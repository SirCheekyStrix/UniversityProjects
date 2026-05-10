"""
token_manager.py — automatyczne zarządzanie JWT z odświeżaniem przez /auth/refresh.

Użycie:
    from token_manager import TokenManager

    tm = TokenManager(
        refresh_url    = "http://localhost:8080/auth/refresh",
        login_url      = "http://localhost:8080/auth/login",   # fallback
        username       = "admin",
        password       = "...",        # lub z env
        token_file     = ".jwt_tokens.json",  # cache tokenów na dysku
    )

    # Pobierz aktualny token (odświeży automatycznie jeśli wygasa)
    headers = tm.headers()
    response = requests.get(url, headers=headers)

    # Jeśli dostaniesz 401 — wymuś odświeżenie i spróbuj ponownie
    if response.status_code == 401:
        headers = tm.headers(force_refresh=True)
        response = requests.get(url, headers=headers)
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import requests

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# KONFIGURACJA — zmień lub ustaw przez env
# ─────────────────────────────────────────────
DEFAULT_LOGIN_URL   = os.environ.get("API_LOGIN_URL",   "http://localhost:8080/auth/login")
DEFAULT_REFRESH_URL = os.environ.get("API_REFRESH_URL", "http://localhost:8080/auth/refresh")
DEFAULT_USERNAME    = os.environ.get("API_USERNAME",    "admin")
DEFAULT_PASSWORD    = os.environ.get("API_PASSWORD",    "")
DEFAULT_TOKEN_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   ".jwt_tokens.json")

# Ile sekund przed wygaśnięciem odświeżamy token (bufor bezpieczeństwa)
REFRESH_BUFFER_SEC  = 60


# ─────────────────────────────────────────────

class TokenManager:
    """
    Thread-safe manager JWT z automatycznym odświeżaniem.

    Priorytet odświeżania:
        1. /auth/refresh z refresh_token  (bezpieczne — bez hasła)
        2. /auth/login z username+password (fallback gdy refresh wygasł)
    """

    def __init__(self,
                 refresh_url: str = DEFAULT_REFRESH_URL,
                 login_url:   str = DEFAULT_LOGIN_URL,
                 username:    str = DEFAULT_USERNAME,
                 password:    str = DEFAULT_PASSWORD,
                 token_file:  str = DEFAULT_TOKEN_FILE):

        self.refresh_url = refresh_url
        self.login_url   = login_url
        self.username    = username
        self.password    = password
        self.token_file  = token_file

        self._lock          = threading.Lock()
        self._access_token:  Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._access_exp:    float = 0.0   # unix timestamp wygaśnięcia
        self._refresh_exp:   float = 0.0

        # Wczytaj zapisane tokeny z dysku (przeżywają restart procesu)
        self._load_from_file()

    # ─── Publiczne API ────────────────────────────────────────────────────────

    def get_token(self, force_refresh: bool = False) -> str:
        """
        Zwraca aktualny access_token.
        Odświeża automatycznie jeśli wygasa za mniej niż REFRESH_BUFFER_SEC.
        """
        with self._lock:
            if force_refresh or self._should_refresh():
                self._do_refresh()
            return self._access_token or ""

    def headers(self, force_refresh: bool = False) -> dict:
        """Zwraca gotowe nagłówki HTTP z Bearer tokenem."""
        return {
            "Authorization": f"Bearer {self.get_token(force_refresh)}",
            "Content-Type":  "application/json",
        }

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Wykonaj request z automatycznym retry po 401.
        Użyj zamiast requests.get/post bezpośrednio.
        """
        kwargs.setdefault("timeout", 15)
        kwargs["headers"] = {**kwargs.get("headers", {}), **self.headers()}

        r = requests.request(method, url, **kwargs)

        # Token wygasł między pobraniem a requestem → odśwież i spróbuj raz jeszcze
        if r.status_code == 401:
            log.warning(f"401 na {url} — odświeżam token i ponawiam")
            kwargs["headers"] = {**kwargs.get("headers", {}),
                                  **self.headers(force_refresh=True)}
            r = requests.request(method, url, **kwargs)

        return r

    def get(self,  url: str, **kwargs) -> requests.Response:
        return self.request("GET",  url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        return self.request("POST", url, **kwargs)

    def is_valid(self) -> bool:
        """Czy aktualny token jest ważny (nie wygasł)."""
        return (self._access_token is not None and
                time.time() < self._access_exp - REFRESH_BUFFER_SEC)

    def token_info(self) -> dict:
        """Zwraca informacje diagnostyczne o aktualnym tokenie."""
        now = time.time()
        return {
            "has_access_token":  self._access_token is not None,
            "has_refresh_token": self._refresh_token is not None,
            "access_expires_in": max(0, round(self._access_exp  - now)),
            "refresh_expires_in":max(0, round(self._refresh_exp - now)),
            "access_valid":      now < self._access_exp  - REFRESH_BUFFER_SEC,
            "refresh_valid":     now < self._refresh_exp - REFRESH_BUFFER_SEC,
        }

    # ─── Wewnętrzne ───────────────────────────────────────────────────────────

    def _should_refresh(self) -> bool:
        """Czy access_token wygasa za mniej niż REFRESH_BUFFER_SEC sekund?"""
        return time.time() >= self._access_exp - REFRESH_BUFFER_SEC

    def _do_refresh(self):
        """Odśwież token — przez refresh_token lub login (fallback)."""

        # Próba 1: /auth/refresh
        if (self._refresh_token and
                time.time() < self._refresh_exp - REFRESH_BUFFER_SEC):
            if self._try_refresh_endpoint():
                return

        # Próba 2: /auth/login (pełne logowanie)
        if self.username and self.password:
            if self._try_login():
                return

        log.error("Nie udało się odświeżyć tokenu — sprawdź credentials")

    def _try_refresh_endpoint(self) -> bool:
        """POST /auth/refresh z refresh_token."""
        try:
            # Próbuj oba typowe formaty payload
            payloads = [
                {"refreshToken": self._refresh_token},
                {"refresh_token": self._refresh_token},
                {"token": self._refresh_token},
            ]
            for payload in payloads:
                r = requests.post(self.refresh_url, json=payload, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    if self._extract_tokens(data):
                        log.info("Token odświeżony przez /auth/refresh")
                        self._save_to_file()
                        return True
                elif r.status_code in (400, 401, 422):
                    continue  # spróbuj inny format
                else:
                    log.warning(f"  /auth/refresh → {r.status_code}")
                    break

        except Exception as e:
            log.warning(f"  /auth/refresh błąd: {e}")
        return False

    def _try_login(self) -> bool:
        """POST /auth/login z username+password."""
        try:
            # Typowe formaty login payload
            payloads = [
                {"username": self.username, "password": self.password},
                {"email":    self.username, "password": self.password},
                {"login":    self.username, "password": self.password},
            ]
            for payload in payloads:
                r = requests.post(self.login_url, json=payload, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    if self._extract_tokens(data):
                        log.info("Token pobrany przez /auth/login")
                        self._save_to_file()
                        return True
                elif r.status_code in (400, 401, 422):
                    continue
                else:
                    log.warning(f"  /auth/login → {r.status_code}: {r.text[:100]}")
                    break

        except Exception as e:
            log.warning(f"  /auth/login błąd: {e}")
        return False

    def _extract_tokens(self, data: dict) -> bool:
        """
        Wyciąga tokeny z odpowiedzi API.
        Obsługuje wiele typowych formatów JWT response.
        """
        # Szukaj access_token pod różnymi kluczami
        access = (data.get("accessToken")
               or data.get("access_token")
               or data.get("token")
               or data.get("jwt"))

        # Szukaj refresh_token
        refresh = (data.get("refreshToken")
                or data.get("refresh_token"))

        if not access:
            log.warning(f"  Brak access_token w odpowiedzi: {list(data.keys())}")
            return False

        self._access_token  = access
        self._access_exp    = self._parse_expiry(access, default_ttl=3600)

        if refresh:
            self._refresh_token = refresh
            self._refresh_exp   = self._parse_expiry(refresh, default_ttl=86400 * 7)

        log.debug(f"  access_token wygasa za "
                  f"{max(0, round(self._access_exp - time.time()))}s")
        return True

    @staticmethod
    def _parse_expiry(jwt_str: str, default_ttl: int = 3600) -> float:
        """
        Dekoduje pole 'exp' z JWT payload (bez weryfikacji podpisu).
        Fallback: teraz + default_ttl sekund.
        """
        try:
            import base64
            parts   = jwt_str.split(".")
            if len(parts) < 2:
                raise ValueError("Nieprawidłowy JWT")
            # Payload: część 2, base64url bez paddingu
            payload = parts[1]
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding
            decoded = json.loads(base64.urlsafe_b64decode(payload))
            exp     = decoded.get("exp")
            if exp:
                return float(exp)
        except Exception as e:
            log.debug(f"  Nie można zdekodować exp z JWT: {e}")
        return time.time() + default_ttl

    # ─── Persystencja ─────────────────────────────────────────────────────────

    def _save_to_file(self):
        """Zapisz tokeny na dysk (przeżywają restart)."""
        try:
            data = {
                "access_token":  self._access_token,
                "refresh_token": self._refresh_token,
                "access_exp":    self._access_exp,
                "refresh_exp":   self._refresh_exp,
                "saved_at":      datetime.now(timezone.utc).isoformat(),
            }
            with open(self.token_file, "w") as f:
                json.dump(data, f, indent=2)
            os.chmod(self.token_file, 0o600)  # tylko właściciel może czytać
        except Exception as e:
            log.warning(f"  Nie można zapisać tokenów: {e}")

    def _load_from_file(self):
        """Wczytaj tokeny z dysku jeśli nie wygasły."""
        if not os.path.exists(self.token_file):
            return
        try:
            with open(self.token_file) as f:
                data = json.load(f)
            now = time.time()
            access_exp  = float(data.get("access_exp",  0))
            refresh_exp = float(data.get("refresh_exp", 0))

            # Wczytaj refresh_token jeśli jeszcze ważny
            if refresh_exp > now + REFRESH_BUFFER_SEC:
                self._refresh_token = data.get("refresh_token")
                self._refresh_exp   = refresh_exp

            # Wczytaj access_token jeśli jeszcze ważny
            if access_exp > now + REFRESH_BUFFER_SEC:
                self._access_token = data.get("access_token")
                self._access_exp   = access_exp
                log.info(f"Token wczytany z pliku "
                         f"(wygasa za {round(access_exp - now)}s)")
            else:
                log.info("Cached access_token wygasł — odświeżę przy pierwszym użyciu")

        except Exception as e:
            log.warning(f"  Błąd wczytywania tokenów: {e}")


# ─────────────────────────────────────────────
# SINGLETON — współdzielony przez wszystkie moduły
# ─────────────────────────────────────────────

_instance: Optional[TokenManager] = None
_instance_lock = threading.Lock()


def get_token_manager(**kwargs) -> TokenManager:
    """
    Zwraca singleton TokenManager.
    Przy pierwszym wywołaniu można podać parametry konfiguracji.
    """
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = TokenManager(**kwargs)
    return _instance


# ─────────────────────────────────────────────
# CLI — diagnostyka i inicjalizacja
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="JWT Token Manager")
    parser.add_argument("--login",    action="store_true", help="Zaloguj i zapisz tokeny")
    parser.add_argument("--refresh",  action="store_true", help="Odśwież token")
    parser.add_argument("--status",   action="store_true", help="Pokaż status tokenów")
    parser.add_argument("--test-url", type=str, default=None,
                        help="Testuj token na podanym URL")
    parser.add_argument("--username", type=str, default=DEFAULT_USERNAME)
    parser.add_argument("--password", type=str, default=DEFAULT_PASSWORD)
    parser.add_argument("--login-url",   type=str, default=DEFAULT_LOGIN_URL)
    parser.add_argument("--refresh-url", type=str, default=DEFAULT_REFRESH_URL)
    args = parser.parse_args()

    tm = TokenManager(
        login_url   = args.login_url,
        refresh_url = args.refresh_url,
        username    = args.username,
        password    = args.password,
    )

    if args.login:
        print("Loguję...")
        if tm._try_login():
            info = tm.token_info()
            print(f"✓ OK — token ważny przez {info['access_expires_in']}s")
            print(f"  refresh_token ważny przez {info['refresh_expires_in']}s")
        else:
            print("✗ Błąd logowania")

    elif args.refresh:
        print("Odświeżam token...")
        token = tm.get_token(force_refresh=True)
        if token:
            info = tm.token_info()
            print(f"✓ OK — token ważny przez {info['access_expires_in']}s")
        else:
            print("✗ Błąd odświeżania")

    elif args.status:
        info = tm.token_info()
        print(f"access_token:   {'✓' if info['has_access_token']  else '✗'} "
              f"(wygasa za {info['access_expires_in']}s, "
              f"{'WAŻNY' if info['access_valid'] else 'WYGASŁ'})")
        print(f"refresh_token:  {'✓' if info['has_refresh_token'] else '✗'} "
              f"(wygasa za {info['refresh_expires_in']}s, "
              f"{'WAŻNY' if info['refresh_valid'] else 'WYGASŁ'})")

    elif args.test_url:
        print(f"Testuję {args.test_url}...")
        r = tm.get(args.test_url)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            print(f"  ✓ OK: {str(r.json())[:200]}")
        else:
            print(f"  ✗ {r.text[:200]}")

    else:
        parser.print_help()
        print("\nPrzykłady:")
        print("  python token_manager.py --login --username admin --password tajne")
        print("  python token_manager.py --status")
        print("  python token_manager.py --refresh")
        print("  python token_manager.py --test-url http://localhost:8080/api/reservoirs")