import pandas as pd

# read matrix
with open("/data/LSY/prostate_cancer/GSE76938_series_matrix.txt", "r") as file:
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

# find status
metadata_df["status"] = metadata_df["Sample_source_name_ch1"].apply(
    lambda x: 1 if "cancer" in x else 0 if "benign" in x else None
)
metadata_df = metadata_df.set_index("Sample_geo_accession")

# read data
data_lines = [line.strip().split() for line in lines if not line.startswith("!")]
data_df = pd.DataFrame(data_lines).T

data_df.columns = data_df.iloc[0]
data_df = data_df.drop(0, axis=0)

data_df = data_df.set_index('"ID_REF"')
data_df = data_df.iloc[:, 1:]
data_df.columns = [col.strip('"') for col in data_df.columns]

data_df = data_df.apply(pd.to_numeric, errors="coerce")

# merge metadata and data
data_df["status"] = metadata_df["status"]

# save data
data_df.to_csv(
    "/data/LSY/prostate_cancer/data_train_and_test.txt",
    index=True,
    header=True,
    sep="\t",
)
