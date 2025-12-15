from pathlib import Path
from charset_normalizer import from_path
import pandas as pd
import csv

class FileValidation :

    def __init__(self, path :str) -> None:
        self.path = path

    def check_file_exists(self) -> bool:
        path = Path(self.path)
        return path.is_file()

    def check_file_readable_as_csv(self) -> bool:
        try:
            encoding = from_path(self.path).best().encoding
            pd.read_csv(self.path, nrows=5, sep=None, engine="python", encoding=encoding)
            return True, f"✅ CSV readable (encoding={encoding})"
        except Exception as e:
            return False, f"❌ CSV not readable: {type(e).__name__}: {e}"

    def check_csv_has_at_east_two_columns(self, delimiter: str | None = None ) -> bool:
        try:
            encoding = from_path(self.path).best().encoding
            df = pd.read_csv(
                self.path,
                nrows=5,
                sep=None,  # let Pandas auto-detect
                engine="python",
                encoding=encoding,
                header=None  # treat first line as data
            )
            n_cols = df.shape[1]
            if n_cols >= 2:
                return True, f"✅ CSV has {n_cols} columns."
            else:
                return False, f"❌ CSV has only {n_cols} column."
        except Exception as e:
            return False, f"❌ CSV not readable: {type(e).__name__}: {e}"

    def check_csv_has_at_least_one_text_field(self) -> bool :
        import pandas as pd
        from charset_normalizer import from_path

        try:
            encoding = from_path(self.path).best().encoding
            df = pd.read_csv(
                self.path,
                nrows=200,
                sep=None,
                engine="python",
                encoding=encoding,
                header=None
            )

            # Iterate through columns
            for col in df.columns:
                series = df[col].dropna().astype(str)
                if len(series) == 0:
                    continue  # skip empty columns

                # Count how many values contain at least one letter
                text_like_count = series.apply(lambda x: any(c.isalpha() for c in x)).sum()
                ratio = text_like_count / len(series)

                if ratio >= 0.7:  # 80%+ of values look like text
                    return True, f"✅ Column {col} is text-like ({ratio * 100:.1f}% of values)."

            return False, "❌ No fully text-like columns detected."

        except Exception as e:
            return False, f"❌ Could not inspect CSV: {type(e).__name__}: {e}"

    def check_csv_has_at_least_one_row(self) -> tuple[bool, str]:
        import pandas as pd
        from charset_normalizer import from_path

        try:
            encoding = from_path(self.path).best().encoding
            df = pd.read_csv(
                self.path,
                nrows=5,  # we only need to know if there's ≥ 1 row
                sep=None,
                engine="python",
                encoding=encoding,
                header=None
            )

            if df.shape[0] > 0:
                return True, f"✅ CSV has {df.shape[0]} data row(s)."
            else:
                return False, "❌ CSV has no data rows (empty file)."

        except pd.errors.EmptyDataError:
            return False, "❌ CSV is empty or unreadable."
        except Exception as e:
            return False, f"❌ Could not inspect CSV: {type(e).__name__}: {e}"

    def validate(self) -> bool:

        # 1️⃣ File existence
        if not self.check_file_exists():
            return False

        # 2️⃣ CSV readability
        if not self.check_file_readable_as_csv():
            return False

        # 3️⃣ At least 2 columns
        if not self.check_csv_has_at_east_two_columns():
            return False

        # 4️⃣ At least one text field
        if not self.check_csv_has_at_least_one_text_field():
            return False

        # 5️⃣ At least one row
        if not self.check_csv_has_at_least_one_row():
            return False

        # ✅ All checks passed
        return True


    # ===============================================================
    #  Smart CSV loader
    # ===============================================================
    def smart_read_csv(self, nrows: int | None = None) -> pd.DataFrame:
        """
        Load a CSV robustly:
        - Never skips the first line (header treated as data if present)
        - Handles quoted and unquoted text
        - Skips malformed lines gracefully
        """
        encoding = from_path(self.path).best().encoding

        # 1️⃣ Try normal quoted CSV read (handles "text, with, commas")
        try:
            df = pd.read_csv(
                self.path,
                sep=",",
                header=None,  # <-- Always treat first line as data
                engine="python",
                encoding=encoding,
                quotechar='"',
                on_bad_lines="skip"
            )
            return df

        # 2️⃣ Fallback: no quoting (handles text without quotes)
        except Exception as e1:
            print(f"⚠️ Quoted read failed ({type(e1).__name__}). Trying unquoted mode...")
            df = pd.read_csv(
                self.path,
                sep=",",
                header=None,  # still no header skip
                engine="python",
                encoding=encoding,
                quoting=csv.QUOTE_NONE,  # ignore quotes completely
                on_bad_lines="skip"
            )
            return df