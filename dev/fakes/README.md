Stand-ins for `marker` and `pypdfium2` used only by `dev/test_e2e.py`, so the real worker.py / pipeline.py / server.py /
md_client.py run end to end on a PC without Marker or GPUs. A fake "PDF" is a text file containing `pages=N`.
