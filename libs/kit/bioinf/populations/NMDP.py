import os
import re
import pandas as pd

from kit.log import log_info, log_caller


NMDP_FOLDER = ""


def load(load_mhc_2=False):
    df_populations = load_populations()

    df_populations_broad = df_populations.groupby("Broad").sum()
    df_populations["frac"] = df_populations.apply(
        lambda row: row.Cnt / df_populations_broad.loc[row.Broad], axis=1
    )

    df_haplotype_freqs = load_haplotype_freqs(load_mhc_2)

    return df_populations, df_populations_broad, df_haplotype_freqs


def load_populations():
    log_caller()
    filename = os.path.join(NMDP_FOLDER, "my_populations.csv")
    log_info(f"load file {filename}")

    return pd.read_csv(filename).set_index("Short")


def load_HLA_ABC():
    log_caller()
    filename = os.path.join(NMDP_FOLDER, "A~C~B.xlsx")
    log_info(f"load file {filename}")

    df = pd.read_excel(filename)
    df["ABC"] = df.apply(
        lambda row: f"{row.A}-{row.B}-{row.C}".replace("*", "")
        .replace("g", "")
        .replace("Q", "")
        .replace("N", ""),
        axis=1,
    )
    df = df.set_index("ABC")
    return df


def trim_hla_name(mhc):
    if mhc in ("DRBX*NNNN"):
        result = "None"
    else:
        h = re.findall(r"^(A|B|C|DRB1|DRB3|DRB4|DRB5|DQB1)\*(\d+):(\d+)[gQNL]*$", mhc)[
            0
        ]
        if h[0] in ("A", "B", "C"):
            result = (
                f"{h[0]}*{h[1]}:{h[2]}".replace("g", "")
                .replace("Q", "")
                .replace("N", "")
                .replace("L", "")
            )
        else:
            result = f"{h[0]}*{h[1]}:{h[2]}".replace("g", "")
    return result


def load_haplotype_freqs(load_mhc_2=False):
    log_caller()

    filename = (
        os.path.join(NMDP_FOLDER, "A~C~B~DRB3-4-5~DRB1~DQB1.xlsx")
        if load_mhc_2
        else os.path.join(NMDP_FOLDER, "A~C~B.xlsx")
    )

    log_info(f"load file {filename}")
    df = pd.read_excel(filename)

    if load_mhc_2:
        df["haplotype"] = df.apply(
            lambda row: f"{trim_hla_name(row.A)}-{trim_hla_name(row.B)}-{trim_hla_name(row.C)}"
            f"-{trim_hla_name(row['DRB3-4-5'])}-{trim_hla_name(row['DRB1'])}-{trim_hla_name(row['DQB1'])}",
            axis=1,
        )
    else:
        df["haplotype"] = df.apply(
            lambda row: f"{trim_hla_name(row.A)}-{trim_hla_name(row.B)}-{trim_hla_name(row.C)}",
            axis=1,
        )

    df = df.set_index("haplotype")
    return df


def get_haplotypes_for_population(df, pop, cover):
    df = df.sort_values(f"{pop}_freq", ascending=False)

    result, covered = [], 0.0
    for idx, row in df.iterrows():
        covered += row[f"{pop}_freq"]
        result.append(idx)
        if covered > cover:
            break

    return result
