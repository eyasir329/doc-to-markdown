## 9. Serve until the deadline or until idle
Checks every 30 s, restarts the API, GPU servers or tunnel if they die, and stops the box after `IDLE_STOP_MIN` minutes without jobs so the weekly GPU quota is not wasted. Stop early with **Cancel run**.
