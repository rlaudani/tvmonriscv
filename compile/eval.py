import plotly.express as px
import pandas as pd

def create_dataframe():
    df = pd.read_csv("compile/timestamps.txt", header=None, names=["layer_id", "runtime"])
    df["iteration"] = df.groupby("layer_id").cumcount() + 1
    df = df[["iteration", "layer_id", "runtime"]]
    return df

def create_boxplots(df, file_path: str = None):
    fig = px.box(df, x="layer_id", y="runtime", log_y=True)
    fig.show()

    if file_path is not None:
        fig.write_image(file_path)
        print(f"Save plot at {file_path}")

if __name__ == "__main__":
    data = create_dataframe()
    create_boxplots(data)
