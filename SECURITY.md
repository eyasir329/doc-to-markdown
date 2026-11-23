# Security policy

## Reporting a vulnerability

Please **don't open a public issue** for a security problem. Report it privately through GitHub instead: the
repository's **Security** tab → **Report a vulnerability**. Include what an attacker could do, the steps to
reproduce it, and the version (release or commit). Maintainers aim to acknowledge reports within a week and will
keep you informed until a fix is released.

## Supported versions

Security fixes go into the latest release. Please check that a problem still exists there before you report it.

## Security model

The box runs in **your own Kaggle session** and is reachable through a public Cloudflare quick-tunnel address
while it runs. What protects it:

| Layer | Protection |
|---|---|
| **Token** | Every endpoint except `GET /ping` needs `Authorization: Bearer <BOX_TOKEN>`. The token must be at least 16 characters and is compared in constant time. The notebook never prints it. At start, the notebook stops if the API answers a request that has no token |
| **Address discovery** | The box posts its tunnel address to an [ntfy.sh](https://ntfy.sh) topic named after a SHA-256 hash of the token, signed with an HMAC of the token. The message holds no secret. Without the token, the topic can't be found and an announcement can't be forged; the client rejects unsigned or wrongly signed announcements |
| **Uploads** | Server-generated job and file IDs. A part is refused as soon as it exceeds the declared size. Total size is capped (`MAX_UPLOAD_GB`). Zip archives are checked for path traversal, file count and unpacked size |
| **Downloads** | Paths are resolved and must stay inside the job's own folder |
| **Isolation** | Each document is converted in a worker process. Results are stored in `/kaggle/tmp`, which isn't saved with the notebook version, and are deleted from the box after download unless you pass `--keep-on-box` |
| **Abandoned jobs** | A job no client has asked about for 15 minutes is cancelled |

## Your responsibilities

- **Use a random token** of 16+ characters (e.g. `python -c "import secrets; print(secrets.token_urlsafe(32))"`), and
  don't reuse a password.
- **Never commit `.box_token`** (it's in `.gitignore`) and never share it. Anyone with the token can use your box
  and its GPU quota while it runs.
- **If the token leaks**, change the Kaggle secret and your `.box_token`, then restart the notebook. Stopping
  the notebook also ends the tunnel address.
- **Know where your documents go:**
  - they travel over HTTPS through Cloudflare's tunnel to your Kaggle session;
  - with the optional `GOOGLE_API_KEY` fallback, page content is sent to Google's Gemini API;
  - without it, nothing else leaves the session.
- **Treat untrusted documents with care.** Converting a document means parsing it with Marker, LibreOffice and
  `ddjvu`. The box runs inside a disposable Kaggle session, but don't keep other secrets in that session.

## Out of scope

Vulnerabilities in Kaggle, Cloudflare, ntfy.sh, Marker, llama.cpp, LibreOffice or the models should be reported
to those projects. Reports about this project's use of them are welcome.
