# Web demo

After the local index and LLM backend are configured, start the demo with:

```bash
./scripts/start_web.sh
```

The launcher defaults to GPU 2, matching `scripts/start_vllm.sh`, and exposes
the service on port 7860. Override the device with
`GPU_ID=0 ./scripts/start_web.sh`. Open <http://127.0.0.1:7860>.

The page calls the same LangGraph pipeline as the CLI and shows the grounded
response, confidence, caveats, source passages, and the four-agent trace.
