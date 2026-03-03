import pandas as pd
import tabulate

def _repr_markdown_(self):
    return f"""
{tabulate.tabulate(
    [
    *self.head(5).itertuples(index=False, name=None),
            ["..."]*(self.shape[1] + 1),
            *self.tail(5).itertuples(index=False, name=None),
        ],
        headers=self.columns,
        tablefmt="pipe",
        showindex=False,
)}

{self.shape[0]} rows x {self.shape[1]} columns
"""
#    return df.head(5).to_markdown(index=False)

pd.DataFrame._repr_markdown_ = _repr_markdown_
del pd.DataFrame._repr_html_
