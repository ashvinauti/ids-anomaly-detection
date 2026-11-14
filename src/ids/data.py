"""Loading the NSL-KDD dataset and mapping attack names to categories.

The NSL-KDD text files have no header row. Each line holds 41 features,
then the attack name (or "normal"), then a difficulty score. This module
gives those columns names, adds a binary target, and adds the broad attack
category (DoS, Probe, R2L, U2R) so results can be broken down later.
"""

from pathlib import Path

import pandas as pd

# Where the dataset files live. data/ is kept out of git.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

# The 41 features, in file order, as described in the NSL-KDD / KDD Cup 99 docs.
FEATURE_COLUMNS: list[str] = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# These three are text values (e.g. "tcp", "http", "SF"), so they need
# one-hot encoding later. Everything else is already a number.
CATEGORICAL_COLUMNS: list[str] = ["protocol_type", "service", "flag"]
NUMERIC_COLUMNS: list[str] = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

# Full column list for reading the raw file: 41 features + label + difficulty.
RAW_COLUMNS: list[str] = FEATURE_COLUMNS + ["attack_name", "difficulty"]

# Map each attack name to one of the four standard categories.
# Names that only appear in KDDTest+ (e.g. mscan, apache2) are included so
# the "unseen attack" records are still categorised correctly.
# Note: a few names (httptunnel, worm) are categorised differently in some
# papers; this follows the most common convention and is documented here.
ATTACK_CATEGORIES: dict[str, str] = {
    # DoS: overwhelm a service so legitimate users cannot reach it.
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "mailbomb": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "worm": "DoS",
    # Probe: scan the network to find hosts, ports or weaknesses.
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L (remote to local): gain access to a machine from outside.
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    "snmpguess": "R2L", "snmpgetattack": "R2L", "httptunnel": "R2L",
    "sendmail": "R2L", "named": "R2L",
    # U2R (user to root): a local user escalates to root privileges.
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def attack_category(attack_name: str) -> str:
    """Return the broad category for one attack name.

    "normal" stays "normal". A name we do not recognise returns "unknown"
    rather than crashing, so a new attack type cannot break the pipeline.
    """
    name = attack_name.strip().lower()
    if name == "normal":
        return "normal"
    return ATTACK_CATEGORIES.get(name, "unknown")


def load_nsl_kdd(path: Path | str) -> pd.DataFrame:
    """Read one NSL-KDD text file and add the target columns.

    Adds two columns:
      - attack_category: normal, DoS, Probe, R2L, U2R (or unknown)
      - is_attack: 0 for normal traffic, 1 for any attack (the binary target)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Download NSL-KDD into the data/ folder first."
        )

    df = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    df["attack_category"] = df["attack_name"].map(attack_category)
    df["is_attack"] = (df["attack_name"] != "normal").astype(int)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the 41 input features (X) from the binary target (y).

    The attack name, category and difficulty are deliberately left out of X:
    giving them to a model would leak the answer.
    """
    return df[FEATURE_COLUMNS].copy(), df["is_attack"].copy()
