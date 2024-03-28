import pandas as pd

inputfilepath2="/data/LSY/gse_matrix_raw/commontrait_Parkinson/GSE111629_series_matrix_2.txt"
inputfilepath="/data/LSY/gse_matrix_raw/commontrait_Parkinson/GSE111629_series_matrix.txt"
outputfilepath= "/data/LSY/finetuneing_cases/commontrait_Parkinson/"
# read matrix
with open(
    inputfilepath2,
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


# find status
metadata_df["status"] = metadata_df.iloc[:, 8].apply(
    lambda x: 1 if "Parkinson" in x else 0 if "control" in x else None
)

metadata_df = metadata_df["status"].apply(pd.to_numeric, errors="coerce")

# read data
data_df = pd.read_table(
    inputfilepath,
    index_col=0,
    header=0,
    delimiter="\t",
)
data_df = data_df.T

data_df = data_df.drop("Unnamed: 573", axis=0)


data_df.columns.name = None
data_df.index = metadata_df.index


# merge metadata and data
data_df["status"] = metadata_df
# drop other disease sample
data_df.dropna(subset=["status"], inplace=True)


from sklearn.model_selection import train_test_split

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
