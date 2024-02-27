import pandas as pd

# read matrix
with open(
    "/data/LSY/commontraits/Kabuki_syndrome/GSE218186_series_matrix_2.txt", "r"
) as file:
    lines = file.readlines()

# find metadata
first_empty_line_index = lines.index("\n")
table_begin_index = next(
    (
        i
        for i, line in enumerate(lines)
        if line.startswith("!series_matrix_table_begin")
    ),
    None,
)
end_index = table_begin_index if table_begin_index is not None else len(lines)
content_before_table = lines[first_empty_line_index:end_index]


# read metadata
metadata_lines = [
    line.strip()[1:].split("\t")
    for line in content_before_table
    if line.startswith("!")
]
metadata_df = pd.DataFrame(metadata_lines).T


metadata_df.columns = metadata_df.iloc[0]
metadata_df = metadata_df.drop(0, axis=0)

metadata_df.dropna(inplace=True)

metadata_df = metadata_df.set_index("Sample_geo_accession")


# find status
metadata_df["status"] = metadata_df.iloc[:, 9].apply(
    lambda x: 1 if "Kabuki" in x else 0 if "Control" in x else None
)


metadata_df.dropna(subset=["status"], inplace=True)
metadata_df = metadata_df["status"].apply(pd.to_numeric, errors="coerce")


# read data
data_df = pd.read_table(
    "/data/LSY/commontraits/Kabuki_syndrome/GSE218186_series_matrix.txt",
    index_col=0,
    header=0,
    delimiter="\t",
)
data_df = data_df.T

data_df = data_df[data_df.index.str.endswith("Beta_value")]


data_df.columns.name = None
data_df.index = metadata_df.index


# merge metadata and data
data_df["status"] = metadata_df
# drop other disease sample
data_df.dropna(subset=["status"], inplace=True)


# save data
data_df.to_csv(
    "/data/LSY/commontraits/Kabuki_syndrome/data_valid.txt",
    index=True,
    header=True,
    sep="\t",
)
