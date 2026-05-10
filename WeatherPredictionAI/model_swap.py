"""
model_swap.py — atomowa podmiana katalogu modelu po zakończeniu treningu.

Zasada:
  1. Trening zapisuje do <MODEL_DIR>_tmp/
  2. Po sukcesie: rename _tmp -> aktywny (atomowe na tym samym FS)
  3. API zawsze czyta z aktywnego katalogu

Użycie w skrypcie treningowym:
    from model_swap import TrainingContext

    with TrainingContext("/path/to/model_files") as ctx:
        # ctx.tmp_dir — katalog do zapisu podczas treningu
        model.save(os.path.join(ctx.tmp_dir, "model.pth"))
    # Po wyjściu z bloku: atomowa podmiana tmp -> aktywny
"""
import os
import shutil
import logging
from contextlib import contextmanager

log = logging.getLogger(__name__)


class TrainingContext:
    """
    Context manager do atomowej podmiany katalogu modelu.

    with TrainingContext("/path/model_files") as ctx:
        # Zapisuj do ctx.tmp_dir
        # Po wyjściu: tmp_dir staje się model_dir atomowo
    """

    def __init__(self, model_dir: str):
        self.model_dir  = os.path.abspath(model_dir)
        self.tmp_dir    = self.model_dir + "_tmp"
        self.backup_dir = self.model_dir + "_backup"

    def __enter__(self):
        # Usuń stary tmp jeśli istnieje (po przerwanym treningu)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)

        # Skopiuj istniejące pliki do tmp (meta.json, config itp.)
        # żeby trening mógł je nadpisać selektywnie
        if os.path.exists(self.model_dir):
            for f in os.listdir(self.model_dir):
                src = os.path.join(self.model_dir, f)
                dst = os.path.join(self.tmp_dir, f)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

        log.info(f"Trening -> {self.tmp_dir}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Błąd podczas treningu — nie podmieniaj, usuń tmp
            log.error(f"Trening nie powiódł się ({exc_type.__name__}) — stary model pozostaje aktywny")
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            return False  # propaguj wyjątek

        # Sukces — atomowa podmiana
        try:
            # Zrób backup starego modelu
            if os.path.exists(self.model_dir):
                if os.path.exists(self.backup_dir):
                    shutil.rmtree(self.backup_dir)
                os.rename(self.model_dir, self.backup_dir)

            # Podmień tmp -> aktywny (atomowe na tym samym FS)
            os.rename(self.tmp_dir, self.model_dir)
            log.info(f"Model podmieniony atomowo: {self.tmp_dir} -> {self.model_dir}")

            # Usuń backup po sukcesie
            if os.path.exists(self.backup_dir):
                shutil.rmtree(self.backup_dir)

        except Exception as e:
            log.error(f"Błąd podmiany modelu: {e}")
            # Przywróć backup jeśli podmiana się nie powiodła
            if os.path.exists(self.backup_dir) and not os.path.exists(self.model_dir):
                os.rename(self.backup_dir, self.model_dir)
                log.info("Przywrócono backup modelu")

        return False


def atomic_save_file(src_path: str, dst_path: str):
    """
    Atomowo zastąp plik dst_path plikiem src_path.
    Używaj gdy podmiana dotyczy pojedynczego pliku.
    """
    tmp_path = dst_path + ".tmp"
    shutil.copy2(src_path, tmp_path)
    os.replace(tmp_path, dst_path)  # os.replace jest atomowe na POSIX