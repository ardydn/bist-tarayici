"""
BIST RSI-30 tarayici (1 saatlik mumlar) -> Telegram bildirimi

Her saat basi (TradingView verisi ~15 dk gecikmeli oldugu icin :16'da) calisir.
Kapanmis son LOOKBACK mumun RSI(14) degerlerine bakar. Bir mumda RSI 30'un
altindan (<=30) ustune (>30) gectiyse, Telegram'a mesaj atar.
Bir calisma atlansa ya da gec kalsa bile kacan kesisimler gecmis mumlardan
bulunur ("N mum once" etiketiyle, tek mesajda toplu gonderilir).
Hacim (VA 20) kontrolu manuel yapilir.

Gun sonu (18:xx calismasi) o gunun ozetini de gonderir.

Kullanim:
    python scanner.py            # normal tarama
    python scanner.py --test     # once Telegram'a test mesaji da gonderir
    python scanner.py --summary  # ozeti simdi gonder (test icin)
Ayni bayraklar RUN_TEST=true / RUN_SUMMARY=true ortam degiskenleriyle de verilebilir.
Gerekli ortam degiskenleri: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from tradingview_screener import Query, col

# ----------------------------------------------------------------------------
# AYARLAR
# ----------------------------------------------------------------------------
MIN_MCAP = 25_000_000_000   # piyasa degeri alt siniri (TL)
RSI_LEVEL = 30.0            # RSI esigi
ATR_MULT = 2.0              # ATR hedef carpani
EXCLUDE = set()             # istemedigin hisseler, ornek: {"ISBTR"}
NOTIFY_DOWN = True          # RSI 30'un altina inenleri de bildir (kapatmak icin False)
MIN_ROWS = 50               # bundan az hisse gelirse veri bozuk say
LOOKBACK = 9                # kesisim aranan kapanmis mum sayisi (~1 islem gunu)
STALE_DAYS = 5              # kayitli RSI bu kadar gunden eskiyse kullanma
KEEP_DAYS = 3               # bildirim kayitlarini kac gun sakla
SCAN_HOURS = range(10, 19)  # beklenen tarama saatleri: 10:16 ... 18:16
SUMMARY_HOUR = 18           # bu saatten sonraki ilk calismada gun ozeti gider
STATE_FILE = Path("state.json")

TZ_TR = timezone(timedelta(hours=3))


# ----------------------------------------------------------------------------
# VERI  ([1] = son kapanmis mum, [2] = ondan bir onceki, ...)
# ----------------------------------------------------------------------------
def _columns(lookback: int) -> list:
    cols = [
        "name",
        "RSI[1]|60",
        "open[1]|60",
        "close[1]|60",
        "ATR[1]|60",
        "BB.upper[1]|60",
        "volume[1]|60",
    ]
    cols += [f"RSI[{k}]|60" for k in range(2, lookback + 1)]
    return cols


def _run_query(cols: list) -> pd.DataFrame:
    q = (
        Query()
        .set_markets("turkey")
        .select(*cols)
        .where(col("market_cap_basic") > MIN_MCAP)
        .order_by("market_cap_basic", ascending=False)
        .limit(500)
    )
    _, df = q.get_scanner_data()
    return df


def fetch() -> pd.DataFrame:
    try:
        return _run_query(_columns(LOOKBACK))
    except Exception as e:
        # gecmis RSI sutunlari reddedilirse eski davranisa (sadece son mum) don
        print("UYARI: gecmis RSI sutunlari alinamadi, sadece son mumla devam:", repr(e))
        return _run_query(_columns(1))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "RSI[1]|60": "rsi",
        "open[1]|60": "open",
        "close[1]|60": "close",
        "ATR[1]|60": "atr",
        "BB.upper[1]|60": "bb_upper",
        "volume[1]|60": "volume",
    }
    rename.update({f"RSI[{k}]|60": f"rsi{k}" for k in range(2, LOOKBACK + 1)})
    d = df.rename(columns=rename)
    d = d.dropna(subset=["rsi", "close", "atr"])
    d = d[d["rsi"] > 0]                                   # verisi bozuk hisseleri ele
    d = d[~d["name"].isin(EXCLUDE)]
    return d.reset_index(drop=True)


def find_crosses(d: pd.DataFrame, prev_rsi: dict) -> pd.DataFrame:
    """
    RSI esigi asagidan yukari kesenler.
    ago = kesisimin kac mum once oldugu (1 = son kapanan mum).
    Gecmis RSI sutunlari yoksa onceki calismanin kaydina gore (ago=1) bakar.
    """
    hist_cols = ["rsi"] + [f"rsi{k}" for k in range(2, LOOKBACK + 1) if f"rsi{k}" in d.columns]
    rows = []  # (satir_no, ago, rsi_onceki, rsi_kesisim_mumu)

    if len(hist_cols) > 1:
        for i, rec in enumerate(d[hist_cols].itertuples(index=False)):
            vals = list(rec)                              # vals[0] = RSI[1] (en yeni)
            for k in range(1, len(vals)):                 # k = kesisim mumu (1 = en yeni)
                cur, prv = vals[k - 1], vals[k]
                if pd.notna(cur) and pd.notna(prv) and prv <= RSI_LEVEL < cur:
                    # eski kesisim sonradan tekrar 30 altina indiyse gecerli sayma
                    if k > 1 and not vals[0] > RSI_LEVEL:
                        break
                    rows.append((i, k, float(prv), float(cur)))
                    break
    else:
        prev = d["name"].map(prev_rsi)                    # kayit yoksa NaN -> False
        for i in d.index[(prev <= RSI_LEVEL) & (d["rsi"] > RSI_LEVEL)]:
            rows.append((i, 1, float(prev[i]), float(d.at[i, "rsi"])))

    cross = d.iloc[[r[0] for r in rows]].copy()
    cross["ago"] = [r[1] for r in rows]
    cross["rsi_prev"] = [r[2] for r in rows]
    cross["rsi_at"] = [r[3] for r in rows]                # kesisim mumunun RSI'si
    cross["atr_hedef"] = cross["close"] + ATR_MULT * cross["atr"]
    cross["key"] = cross["name"] + "|" + cross["rsi_at"].round(4).astype(str)
    return cross


def find_down_crosses(d: pd.DataFrame, prev_rsi: dict) -> pd.DataFrame:
    """RSI esigin altina inenler (onceki >= esik, simdiki < esik)."""
    prev = d["name"].map(prev_rsi)
    down = d[(prev >= RSI_LEVEL) & (d["rsi"] < RSI_LEVEL)].copy()
    down["rsi_prev"] = prev[down.index]
    down["key"] = "dn|" + down["name"] + "|" + down["rsi"].round(4).astype(str)
    return down


# ----------------------------------------------------------------------------
# DURUM (onceki RSI degerleri, tekrar bildirimi engeli, gunluk kayit)
# ----------------------------------------------------------------------------
def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict) -> None:
    limit = datetime.now(TZ_TR) - timedelta(days=KEEP_DAYS)
    state["sent"] = {
        k: v for k, v in state.get("sent", {}).items()
        if datetime.fromisoformat(v) > limit
    }
    STATE_FILE.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")


def previous_rsi(state: dict, now: datetime) -> dict:
    try:
        age = now - datetime.fromisoformat(state["updated"])
        if age > timedelta(days=STALE_DAYS):
            return {}
        return state.get("rsi", {})
    except Exception:
        return {}


def today_log(state: dict, now: datetime) -> dict:
    """Bugunun kaydi; gun degistiyse sifirlanir."""
    date = now.strftime("%Y-%m-%d")
    if state.get("today", {}).get("date") != date:
        state["today"] = {"date": date, "scans": [], "crosses": []}
    return state["today"]


# ----------------------------------------------------------------------------
# TELEGRAM
# ----------------------------------------------------------------------------
def send(text: str) -> None:
    token = os.environ["TELEGRAM_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        },
        timeout=30,
    )
    r.raise_for_status()


def build_message(r) -> str:
    link = f"https://www.tradingview.com/chart/?symbol=BIST:{r.name}"
    if pd.isna(r.open):
        renk = "?"
    else:
        renk = "yeşil" if r.close >= r.open else "kırmızı"
    hacim = "?" if pd.isna(r.volume) else f"{r.volume:,.0f}".replace(",", ".")
    bb = "?" if pd.isna(r.bb_upper) else f"{r.bb_upper:.2f}"
    return (
        f"🟡 <b>{r.name}</b> — 1S mum kapandı, RSI {RSI_LEVEL:.0f}'u yukarı kesti\n"
        f"RSI: {r.rsi_prev:.1f} → {r.rsi:.1f}\n"
        f"M1 kapanış: {r.close:.2f} ({renk} mum)\n"
        f"M1 hacim: {hacim}\n"
        f"ATR(2x) hedef: {r.atr_hedef:.2f}\n"
        f"Bollinger üst: {bb}\n\n"
        f"Kontrol et: hacim barı yeşil ve VA(20) üstünde mi? "
        f"Sonraki saatte M2 onayına bak.\n"
        f"{link}"
    )


def build_late_message(df: pd.DataFrame) -> str:
    lines = [
        f"🟠 <b>Geç yakalanan RSI {RSI_LEVEL:.0f} yukarı kesişimleri</b>",
        "(önceki tarama atlandı/gecikti; giriş penceresi geçmiş olabilir, grafiğe bak)",
        "",
    ]
    for r in df.sort_values("ago").itertuples():
        lines.append(
            f"• <b>{r.name}</b> — {r.ago} mum önce — RSI {r.rsi_prev:.1f} → {r.rsi_at:.1f} "
            f"(şu an {r.rsi:.1f})"
        )
    return "\n".join(lines)


def build_down_message(df: pd.DataFrame) -> str:
    lines = [f"🔻 <b>RSI {RSI_LEVEL:.0f}'un altına inenler</b> (1S mum kapandı)", ""]
    for r in df.itertuples():
        lines.append(f"• <b>{r.name}</b> — RSI {r.rsi_prev:.1f} → {r.rsi:.1f} — {r.close:.2f}")
    lines.append("")
    lines.append("Yukarı kesişim için izleme listesine al.")
    return "\n".join(lines)


def build_summary(day: dict, now: datetime) -> str:
    lines = [f"📋 <b>Günün özeti — {now:%d.%m.%Y}</b>", ""]

    crosses = day["crosses"]
    if crosses:
        lines.append(f"RSI 30'u yukarı kesen hisseler ({len(crosses)}):")
        for c in crosses:
            geç = f" ({c['ago']} mum önce)" if c.get("ago", 1) > 1 else ""
            lines.append(
                f"• <b>{c['name']}</b> — {c['time']}{geç} — RSI {c['rsi_prev']:.1f} → "
                f"{c['rsi']:.1f} — {c['close']:.2f}"
            )
    else:
        lines.append(f"Bugün RSI {RSI_LEVEL:.0f}'u yukarı kesen hisse olmadı.")

    if NOTIFY_DOWN:
        downs = day.get("downs", [])
        lines.append("")
        if downs:
            lines.append(f"RSI {RSI_LEVEL:.0f}'un altına inenler ({len(downs)}):")
            for c in downs:
                lines.append(
                    f"• <b>{c['name']}</b> — {c['time']} — RSI {c['rsi_prev']:.1f} → {c['rsi']:.1f}"
                )
        else:
            lines.append(f"Bugün RSI {RSI_LEVEL:.0f}'un altına inen hisse olmadı.")

    expected = [h for h in SCAN_HOURS if h <= now.hour]
    done = set(day["scans"])
    missing = [h for h in expected if h not in done]
    lines.append("")
    lines.append(f"Tarama: {len(expected) - len(missing)}/{len(expected)}")
    if missing:
        lines.append("Çalışmayan saatler: " + ", ".join(f"{h}:16" for h in missing))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# ANA AKIS
# ----------------------------------------------------------------------------
def main(now: datetime = None) -> None:
    now = now or datetime.now(TZ_TR)
    want_test = "--test" in sys.argv or os.environ.get("RUN_TEST") == "true"
    want_summary = "--summary" in sys.argv or os.environ.get("RUN_SUMMARY") == "true"

    if want_test:
        send(f"✅ Tarayıcı çalışıyor. Test mesajı ({now:%d.%m.%Y %H:%M}).")

    d = prepare(fetch())
    if len(d) < MIN_ROWS:
        print(f"HATA: sadece {len(d)} hisse geldi, veri bozuk gorunuyor. Durum guncellenmedi.")
        sys.exit(1)

    state = load_state()
    prev = previous_rsi(state, now)
    crosses = find_crosses(d, prev)

    sent = state.setdefault("sent", {})
    day = today_log(state, now)
    new = crosses[~crosses["key"].isin(sent)]
    fresh = new[new["ago"] == 1]
    late = new[new["ago"] > 1]

    has_hist = "rsi2" in d.columns
    print(
        f"[{now:%Y-%m-%d %H:%M}] taranan: {len(d)} | gecmis RSI: {'var' if has_hist else 'YOK'} | "
        f"kesisim: {len(crosses)} | yeni: {len(new)} (taze {len(fresh)}, gec {len(late)})"
    )
    if NOTIFY_DOWN and not prev:
        print("  Not: onceki RSI kaydi yok (ilk calisma veya eski kayit), asagi kesen aranmadi.")

    for r in fresh.itertuples():
        try:
            send(build_message(r))
            sent[r.key] = now.isoformat()
            day["crosses"].append(
                {
                    "name": r.name,
                    "time": f"{now:%H:%M}",
                    "ago": 1,
                    "rsi_prev": float(r.rsi_prev),
                    "rsi": float(r.rsi),
                    "close": float(r.close),
                }
            )
            print("  gonderildi:", r.name)
        except Exception as e:  # gonderilemezse sonraki calismada tekrar denenir
            print("  GONDERILEMEDI:", r.name, repr(e))

    if len(late):
        try:
            send(build_late_message(late))
            for r in late.itertuples():
                sent[r.key] = now.isoformat()
                day["crosses"].append(
                    {
                        "name": r.name,
                        "time": f"{now:%H:%M}",
                        "ago": int(r.ago),
                        "rsi_prev": float(r.rsi_prev),
                        "rsi": float(r.rsi_at),
                        "close": float(r.close),
                    }
                )
            print(f"  gec yakalanan toplu mesaj gonderildi: {len(late)} hisse")
        except Exception as e:
            print("  GEC YAKALANAN MESAJ GONDERILEMEDI:", repr(e))

    if NOTIFY_DOWN:
        downs = find_down_crosses(d, prev)
        downs = downs[~downs["key"].isin(sent)]
        if len(downs):
            try:
                send(build_down_message(downs))
                for r in downs.itertuples():
                    sent[r.key] = now.isoformat()
                    day.setdefault("downs", []).append(
                        {
                            "name": r.name,
                            "time": f"{now:%H:%M}",
                            "rsi_prev": float(r.rsi_prev),
                            "rsi": float(r.rsi),
                        }
                    )
                print(f"  asagi kesen bildirimi gonderildi: {len(downs)} hisse")
            except Exception as e:
                print("  ASAGI KESEN BILDIRIMI GONDERILEMEDI:", repr(e))

    state["rsi"] = {n: float(v) for n, v in zip(d["name"], d["rsi"])}
    state["updated"] = now.isoformat()
    day["scans"].append(now.hour)

    auto_summary = now.hour >= SUMMARY_HOUR and state.get("summary_date") != day["date"]
    if want_summary or auto_summary:
        try:
            send(build_summary(day, now))
            if auto_summary:
                state["summary_date"] = day["date"]
            print("  gun ozeti gonderildi")
        except Exception as e:
            print("  OZET GONDERILEMEDI:", repr(e))

    save_state(state)


if __name__ == "__main__":
    main()
