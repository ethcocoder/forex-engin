# Historical FX Data Source Findings

## Primary source: Dukascopy

The existing downloader targets the documented Dukascopy historical archive. Its archive format supplied a valid 2024 EUR/USD sample in the prior implementation pass, but the 2020 campaign start encountered SSL handshake timeouts and a truncated archive. The downloader was hardened to delete and retry malformed cache entries. The source is therefore retained as the primary full-tick path, but the multi-year campaign is not marked complete while the source is temporarily unreachable from this environment.

## Disclosed fallback candidate: HistData Generic ASCII tick data

HistData publishes Generic ASCII tick data for EUR/USD and other pairs. Its FAQ states that Generic ASCII tick files contain `DateTime,Bid,Ask,Volume`, include Ask prices for spread calculation, are organized by pair/year/month, and use EST without daylight-saving adjustments. It also disclaims warranty/certification and provides gap statistics per file. Its NinjaTrader tick offering is documented as one-second resolution.

Implications: HistData may be used only as a **clearly labelled secondary research source** after a time-zone conversion and gap audit. It must not be silently combined with Dukascopy data or represented as a substitute for full millisecond tick data in high-frequency execution validation.

## References

[1] https://www.dukascopy.com/swiss/english/marketwatch/historical/
[2] https://www.dukascopy.com/wiki/en/development/strategy-api/historical-data/history-ticks/
[3] https://www.histdata.com/download-free-forex-data/?/ascii/tick-data-quotes
[4] https://www.histdata.com/f-a-q/

## Verified public catalogue route

Browser inspection confirmed this public navigation sequence for EUR/USD Generic ASCII tick data:

1. `https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/EURUSD`
2. `https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/eurusd/2020`
3. `https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/eurusd/2020/1`

The final page is expected to expose the specific January 2020 ZIP archive. The source must be labelled secondary, time-zone-normalised from its documented EST-without-DST basis, and subjected to gap auditing before any research use.

## January 2020 archive verification

The public January 2020 EUR/USD Generic ASCII tick page was opened at:

`https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/eurusd/2020/1`

It advertises the file `HISTDATA_COM_ASCII_EURUSD_T_202001.zip` and a companion status file `HISTDATA_COM_ASCII_EURUSD_T_202001.txt`. This page identifies the data as Generic CSV, EUR/USD, Tick Data, year/month 202001.
