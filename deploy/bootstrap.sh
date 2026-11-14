#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu server (run as root): Docker, firewall, fail2ban.
#   ssh root@<ip> 'bash -s' < deploy/bootstrap.sh
set -euo pipefail

if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
apt-get install -y -q ufw fail2ban rsync

# Only SSH and the web ports. Docker-published ports bypass ufw, so compose.prod.yaml publishes only Caddy.
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 443/udp
ufw --force enable
systemctl enable --now fail2ban

mkdir -p /srv/stoa
IP=$(curl -fsS4 https://api.ipify.org || hostname -I | awk '{print $1}')
echo "Ready. Public IP: $IP -> SITE_ADDRESS=${IP//./-}.sslip.io"
