## 5. Box files (these cells only write files)
`convert.py` routes each file type (LibreOffice first for office files, direct conversion for txt/md/csv); `lb.py` balances VLM requests over the GPU servers; `worker.py` is a warm Marker worker; `pipeline.py` runs one job; `server.py` is the API.
