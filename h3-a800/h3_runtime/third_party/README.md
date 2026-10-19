# third_party

The Sol-Attn kernel package used by `--attention-backend sol` is not redistributed.
Fetch it from [NVlabs/Sana](https://github.com/NVlabs/Sana) (branch `sol-engine`, pinned commit) with

```bash
bash fetch_sol_attn.sh        # from h3-a800/; creates h3_runtime/third_party/sol_attn/
```

or point `H3_SOL_ATTN_ROOT` at a directory that contains `sol_attn/`. The `dense` and `vsa` backends do not need it.
