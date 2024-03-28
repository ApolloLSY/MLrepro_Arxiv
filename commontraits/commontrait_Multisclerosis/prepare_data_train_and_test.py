import pandas as pd
from sklearn.model_selection import train_test_split

inputfilepath = "/data/LSY/z_preparing_and_parts/commontrait_Multisclerosis/GSE106648_series_matrix.txt"
outputfilepath = "/data/LSY/z_preparing_and_parts/commontrait_Multisclerosis/"


with open(
    inputfilepath,
    "r",
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


mask = metadata_df.iloc[:, 11] != '"age: 0"'

metadata_df = metadata_df[mask]


# find status
metadata_df["status"] = metadata_df.iloc[:, 9].apply(
    lambda x: 1 if "MS" in x else 0 if "control" in x else None
)


metadata_df = metadata_df["status"].apply(pd.to_numeric, errors="coerce")


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
data_df["status"] = metadata_df
# drop other disease sample
data_df.dropna(subset=["status"], inplace=True)


train_data, test_data = train_test_split(data_df, test_size=0.2, random_state=42)


# save data
train_data.to_csv(
    outputfilepath+"data_train.txt",
    index=True,
    header=True,
    sep="\t",
)


# save data
test_data.to_csv(
    outputfilepath+"data_test.txt",
    index=True,
    header=True,
    sep="\t",
)
