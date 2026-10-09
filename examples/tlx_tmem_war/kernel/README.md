# Affected kernel source

`blackwell_fa_ws_pipelined_persistent.py` is the public source from
[`facebookexperimental/triton@d4e09c9e`](https://github.com/facebookexperimental/triton/commit/d4e09c9e8c9c1ba3b1dbc974756b32d613e94942):

```text
third_party/tlx/tutorials/blackwell_fa_ws_pipelined_persistent.py
```

The file is bundled so that `../repro.py` does not require a second historical
checkout. It still requires a current fbtriton installation with TLX support.
