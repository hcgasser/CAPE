""" This module contains the CAPE database class."""


import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from kit.loch import file_to_seq_hashes
from kit.loch.utils import get_seq_hash
from kit.loch.path import (
    get_pdb_file_path,
    get_fasta_file_path,
    get_md_path,
    get_function_path,
)
from kit.bioinf.fasta import read_fasta
from kit.bioinf.immuno.utils import to_mhc_prefix
from kit.bioinf.alignment.structure.tm_align import align_structures
from kit.data.db import BasicDB

from CAPE.utils import get_seq_set_hash


class CapeDB(BasicDB):
    """This class provides access to the CAPE database. It is a subclass of BasicDB.
    It organizes the data for the evaluation of the CAPE designs

    :param database_path: path to the database
    :param domain: domain of the designs
    :param loch_path: path to the protein information
    :param predictors_mhc_1: dictionary of MHC-I predictors
    :param predictor_structure_name: name of the structure predictor
    :param predictor_function_name: name of the function predictor
    :param pairwise_sequence_aligner: pairwise sequence aligner
    :param create_database: whether or not to create a new database
    """

    def __init__(
        self,
        database_path,
        domain,
        loch_path,
        predictors_mhc_1=None,
        predictor_structure_name="AF",
        predictor_function_name="TransFun",
        pairwise_sequence_aligner=None,
        create_database=False,
    ):
        super().__init__(database_path, connect=False)
        self.domain = domain
        self.predictors_mhc_1 = predictors_mhc_1
        self.predictor_structure_name = predictor_structure_name
        self.predictor_function_name = predictor_function_name
        self.pairwise_sequence_aligner = pairwise_sequence_aligner

        self.n_supports = 300

        # directories of the protein information
        self.loch_path = loch_path
        if create_database:
            if os.path.exists(database_path):
                os.remove(database_path)
            self.connect()
            self.create_database()
        else:
            self.connect()

    def create_database(self):
        """Creates the database"""

        cursor = self.cursor

        # Retrieve table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # Drop tables
        for table in tables:
            table_name = table[0]
            cursor.execute(f"DROP TABLE IF EXISTS {table_name};")

        cursor.execute(
            """
            CREATE TABLE sequences (
                seq_hash TEXT PRIMARY KEY,
                seq TEXT,
                complete BOOLEAN,
                Energy_ESM REAL,
                Energy_AF REAL,
                UNIQUE(seq)
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE packs (
                domain TEXT,
                pack TEXT,
                seq_hash TEXT,
                UNIQUE(domain, pack, seq_hash),
                FOREIGN KEY (seq_hash) REFERENCES sequences(seq_hash)
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE representations (
                domain TEXT,
                seq_hash TEXT,
                rep_name TEXT,
                rep_base_hash TEXT,
                rep TEXT,
                UNIQUE(domain, seq_hash, rep_name, rep_base_hash),
                FOREIGN KEY (seq_hash) REFERENCES sequences (seq_hash)
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE alignments_structure (
                seq_hash_from TEXT,
                seq_hash_to TEXT,
                TMscore REAL,
                aligned_length INTEGER,
                rmsd REAL,
                identical INTEGER,
                UNIQUE(seq_hash_from, seq_hash_to),
                FOREIGN KEY (seq_hash_from) REFERENCES data(seq_hash),
                FOREIGN KEY (seq_hash_to) REFERENCES sequences(seq_hash)
            )
        """
        )
        self.conn.commit()

    def get_pdb_file_path(self, seq_hash):
        """For a given sequence hash, returns the path to the corresponding PDB file"""

        return get_pdb_file_path(
            seq_hash, self.loch_path, self.predictor_structure_name
        )

    def get_fasta_file_path(self, seq_hash):
        """For a given sequence hash, returns the path to the corresponding FASTA file"""

        return get_fasta_file_path(seq_hash, self.loch_path)

    def get_md_path(self, seq_hash, md_param_hash):
        """For a given sequence hash, returns the path to the corresponding MD directory"""

        return get_md_path(
            seq_hash, md_param_hash, self.loch_path, self.predictor_structure_name
        )

    def get_function_path(self, seq_hash):
        """For a given sequence hash, returns the path to the corresponding function file"""

        return get_function_path(
            seq_hash,
            self.loch_path,
            self.predictor_structure_name,
            self.predictor_function_name,
        )

    def get_sequence(self, seq=None, seq_hash=None):
        """Returns the row of the sequence table corresponding to the given sequence or hash code.
        If seq is not present, a new entry is created.

        :param seq: sequence to be inserted
        :param seq_hash: hash code of the sequence
        :return: hash code of the sequence
        """

        if seq is None and seq_hash is None:
            raise ValueError("Either seq or seq_hash must be specified")
        if seq is not None and seq_hash is not None:
            raise ValueError("Only one of seq or seq_hash must be specified")

        if seq is not None:
            seq = seq.translate(str.maketrans("", "", "*-"))
            seq_hash = get_seq_hash(seq)

        self.cursor.execute(f"SELECT * FROM sequences WHERE seq_hash = '{seq_hash}'")
        result = self.cursor.fetchone()
        if result is None:
            sql = """
                INSERT INTO sequences (seq, seq_hash, complete)
                    VALUES (?, ?, ?)
                """
            self.cursor.execute(sql, (seq, seq_hash, "X" not in seq))
            result = self.get_sequence(seq)
        else:
            self.cursor.execute(
                f"SELECT * FROM sequences WHERE seq_hash = '{seq_hash}'"
            )
            result = self.cursor.fetchone()
            if result is None:
                raise ValueError(
                    f"Sequence with hash {seq_hash} does not exist in the database"
                )
        return result

    def add_seq_hashes_as_pack(self, seq_hash_file_path):
        """Adds all sequence hashes in a file as a pack in the database"""

        if os.path.exists(seq_hash_file_path):
            pack_name = (
                os.path.basename(seq_hash_file_path)
                .removesuffix(".seq_hash")
                .removeprefix(f"{self.domain}.")
            )
            print(pack_name)

            seq_hashes = file_to_seq_hashes(seq_hash_file_path)

            for seq_hash in tqdm(seq_hashes):
                seq = list(read_fasta(self.get_fasta_file_path(seq_hash)))[0]
                sequence = self.get_sequence(seq=seq)
                self.add_sequence_to_pack(sequence, pack_name)
            self.conn.commit()

    def add_sequence_to_pack(self, sequence, pack):
        """Adds a sequence to a pack

        :param sequence: sequence to be inserted
        :param pack: the set used to generate the sequence
        :return: None
        """

        seq_hash = sequence["seq_hash"]

        df = self.sql_to_df(
            f"""
            SELECT * FROM packs
            WHERE domain == '{self.domain}' and pack == '{pack}' and seq_hash == '{seq_hash}'
        """
        )
        if df.empty:
            sql = """
                INSERT INTO packs (domain, pack, seq_hash) 
                VALUES (?, ?, ?)
            """
            self.cursor.execute(sql, (self.domain, pack, seq_hash))

    def get_pack(self, pack, complete=True):
        """Returns all the information stored in the sequences table for the sequences in a pack"""

        sql = f"""
            SELECT s.*
            FROM packs p INNER JOIN sequences s ON p.seq_hash == s.seq_hash WHERE p.domain == '{self.domain}' AND p.pack == '{pack}'
        """
        if complete:
            sql += " AND complete == 1"
        return self.sql_to_df(sql)

    def get_closest(self, seq, self_match):
        """Returns the closest support hash for the given sequence

        :param seq: sequence to be queried
        :param self_match: whether or not to match the sequence with itself
        """

        sequence = self.get_sequence(seq=seq)
        seq_hash_to, seq = sequence["seq_hash"], sequence["seq"]

        max_tmscore_support_seq_hash = None
        max_tmscore = -1.0
        max_tmscore_aligned_length = -1
        max_tmscore_rmsd = -1

        df_supports = self.get_pack("support")

        for _, row in df_supports.iterrows():
            support_seq_hash = row["seq_hash"]

            if support_seq_hash != seq_hash_to or self_match:
                df = self.get_TMalignment(support_seq_hash, seq_hash_to)

                support_seq_hash = df.iloc[0].seq_hash_from
                tmscore = df.iloc[0].TMscore
                aligned_length = df.iloc[0].aligned_length
                rmsd = df.iloc[0].rmsd
                if tmscore > max_tmscore:
                    max_tmscore_support_seq_hash = support_seq_hash
                    max_tmscore = tmscore
                    max_tmscore_aligned_length = aligned_length
                    max_tmscore_rmsd = rmsd

        return (
            max_tmscore_support_seq_hash,
            max_tmscore,
            max_tmscore_aligned_length,
            max_tmscore_rmsd,
        )

    def get_TMalignment(self, seq_hash_from, seq_hash_to):
        """Returns the TM alignment information between two sequences"""

        from_file_path = self.get_pdb_file_path(seq_hash_from)
        to_file_path = self.get_pdb_file_path(seq_hash_to)

        df = self.sql_to_df(
            f"""
            SELECT * FROM alignments_structure
            WHERE seq_hash_from == '{seq_hash_from}' AND seq_hash_to == '{seq_hash_to}'
        """
        )

        if len(df) == 0:
            tm_score, aligned_length, rmsd, identical = align_structures(
                from_file_path, to_file_path
            )

            sql = """
                INSERT INTO alignments_structure (seq_hash_from, seq_hash_to, 
                            TMscore, aligned_length, rmsd, identical)
                VALUES (?, ?, ?, ?, ?, ?)
            """

            self.cursor.execute(
                sql,
                (seq_hash_from, seq_hash_to, tm_score, aligned_length, rmsd, identical),
            )
            self.conn.commit()
            return self.get_TMalignment(seq_hash_from, seq_hash_to)
        return df

    def add_mhc_1_presentations(self, predictor_mhc_1_name, allele):
        """For all sequences in the database for which no MHC Class 1
        presentation prediction is stored yet, adds this information
        into the database

        :param predictor_mhc_1_name: name of the MHC-I predictor
        :param allele: MHC-I allele
        """

        table = predictor_mhc_1_name
        if not self.exists_table(table):
            self.cursor.execute(
                f"""
               CREATE TABLE {table} (
                   seq_hash TEXT PRIMARY KEY,
                   FOREIGN KEY (seq_hash) REFERENCES sequences (seq_hash)
                )
            """
            )

        column = to_mhc_prefix(allele)
        self.exists_column(table, column, add="INTEGER DEFAULT NULL")

        # Retrieve all sequences that have not been assessed for the allele yet
        df = self.sql_to_df(
            f"""
            SELECT a.seq_hash, a.seq 
            FROM sequences a LEFT JOIN {table} b ON a.seq_hash == b.seq_hash
            WHERE b.{column} IS NULL AND a.complete == 1
        """
        )
        for _, row in df.iterrows():
            df_2 = self.sql_to_df(
                f"""SELECT * FROM {table} WHERE seq_hash = '{row.seq_hash}'"""
            )
            if df_2.empty:
                sql = f"""
                    INSERT INTO {table} ({column}, seq_hash)
                    VALUES (?, ?)
                """
            else:
                sql = f"""UPDATE {table} SET {column} = ? WHERE seq_hash = ? """
            visibility = len(
                self.predictors_mhc_1[predictor_mhc_1_name].seq_presented(
                    row.seq, [allele]
                )
            )
            self.cursor.execute(sql, (visibility, row.seq_hash))

    def get_visibility_mhc_1(self, predictor_mhc_1_name, mhc_1_alleles, seq_hash):
        """Using the information stored in the database, it returns the visibility of a sequence

        :param predictor_mhc_1_name: name of the MHC-I predictor
        :param mhc_1_alleles: list of MHC-I alleles to get the visibility for
        :param seq_hash: hash code of the sequence
        :return: visibility of the sequence
        """

        table = predictor_mhc_1_name
        columns = ", ".join(
            [to_mhc_prefix(mhc_1_allele) for mhc_1_allele in mhc_1_alleles]
        )

        sql = f"""
            SELECT {columns}
            FROM {table}
            WHERE seq_hash == '{seq_hash}'
        """
        self.cursor.execute(sql)
        visibility = self.cursor.fetchone()
        return np.sum(visibility)

    def get_missing_rep(self, rep_name):
        """Returns all sequences for which no representation is stored yet"""

        return self.sql_to_df(
            f"""
            SELECT s.* 
            FROM sequences s LEFT JOIN 
                (SELECT * 
                 FROM representations a
                 WHERE a.domain == '{self.domain}' AND a.rep_name == '{rep_name}'
                 ) r ON s.seq_hash == r.seq_hash
            WHERE s.complete == 1 AND r.rep IS NULL
        """
        )

    def add_support_rep(self):
        supports = self.sql_to_df(
            f"""
            SELECT s.* 
            FROM packs p LEFT JOIN sequences s ON p.seq_hash == s.seq_hash
            WHERE p.domain == '{self.domain}' AND p.pack == 'support' AND s.complete == 1
            ORDER BY p.seq_hash ASC
        """
        ).seq
        sql = """
            INSERT INTO representations (domain, seq_hash, rep_name, rep)
            VALUES (?, ?, ?, ?)
        """
        df = self.get_missing_rep("support")
        for _, row in tqdm(df.iterrows(), desc="missing representations"):
            seq = row["seq"]
            separations = self.pairwise_sequence_aligner.get_seq_to_seqs_dissimilarity(
                seq, supports
            )
            representation = ",".join([f"{r}" for r in separations])
            self.cursor.execute(
                sql, (self.domain, row["seq_hash"], "support", representation)
            )

        self.conn.commit()

    def get_rep(self, rep_name, rep_base_hash=None):
        sql = f"""
            SELECT r.seq_hash, r.rep
            FROM representations r
            WHERE r.domain == '{self.domain}' AND r.rep_name == '{rep_name}'
        """
        if rep_base_hash is not None:
            sql += f" AND r.rep_base_hash == '{rep_base_hash}' "

        df = self.sql_to_df(sql).set_index("seq_hash")
        df_aux = pd.DataFrame(
            index=df.index,
            data=np.array(df.rep.str.split(",", expand=True).astype(float)),
        )
        df_aux.columns = [f"rep_{i + 1}" for i in range(df_aux.shape[1])]
        df = df.join(df_aux, how="left")
        df.drop(columns=["rep"], inplace=True)
        return df

    def add_reduced_support_rep(self, df_eval, rep_name, method):
        rep_base_hash = get_seq_set_hash(list(set(df_eval["seq_hash"])))

        self.cursor.execute(
            f"""
            DELETE FROM representations
            WHERE domain == '{self.domain}' AND rep_name == '{rep_name}' AND rep_base_hash == '{rep_base_hash}'
        """
        )

        df = self.get_rep("support")
        df = (
            df_eval[["pack", "seq_hash"]]
            .join(df, how="left", on="seq_hash")
            .drop(columns=["pack"])
            .drop_duplicates(subset=["seq_hash"])
        )

        result = method.fit_transform(
            df[[f"rep_{c+1}" for c in range(self.n_supports)]]
        )

        sql = """
            INSERT INTO representations (domain, seq_hash, rep_name, rep_base_hash, rep)
            VALUES (?, ?, ?, ?, ?)
        """
        added = set()
        assert len(df) == len(result)
        for idx, res in enumerate(result):
            seq_hash = df.iloc[idx].seq_hash
            if seq_hash not in added:
                self.cursor.execute(
                    sql,
                    (
                        self.domain,
                        seq_hash,
                        rep_name,
                        rep_base_hash,
                        ",".join([str(r) for r in res]),
                    ),
                )
                added.add(seq_hash)

        self.conn.commit()
        return rep_base_hash
