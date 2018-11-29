"""
Basic analysis to understand what exactly was sent by Frangi.

"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
from util import add_identity_lines

def devnull(*args,**kwargs):
    return


def get_records_with_manual(fn,verbose = True):
    p = print if verbose else devnull
    # Read
    dfs = [pd.read_excel(f) for f in fn]
    s = [set(df["ID"]) for df in dfs]
    common_patients = s[0].intersection(s[1])
    for i in range(len(fn)):
        p("File {} contains {} patient IDs".format(fn[i],len(s[i])))
    p("There are {} patient IDs common to both excel files".format(len(common_patients)))

    # attempting a database-style inner join
    df = pd.merge(dfs[0],dfs[1],how="inner",on=['ID'])
    df["ID"].isin(common_patients).sum()
    p("Inner join on ID gives {} records".format(len(df)))
    cd = df.drop_duplicates()
    p("Of which {} are unique".format(len(cd)))
    c = cd.drop_duplicates(subset=["ID"])
    p("Of which {} have unique IDs".format(len(c)))
    d = c.dropna()
    p("Of which {} rows don't contain NaNs".format(len(d)))
    p("......................................")
    return c


def get_biomarker_tokens(df):
    cs = list(df.columns) # cast to list just to be safe that I don't damage the df
    ts = dict() # token structure
    for cn in cs:
        c = cn.split('_')
        if len(c)<2:
            continue
        t = c[0] # token
        k = c[1] # kind
        if k not in ('manual','automatic'):
            continue
        if t in ts:
            if k in ts[t]:
                ts[t][k].append(cn)
            else:
                ts[t][k] = [cn]
        else:
            ts[t] = {k:[cn]}
    return ts


def plot_auto_vs_manual(df):
    """
    Take a dataframe and plot columns with names containing the same first letters
    before an underscore character (the "token") and "automatic" after the first
    underscore as ordinate and "manual" after the first underscore as abscissa
    """
    ts = get_biomarker_tokens(df)
    # Iterate over tokens
    for t in ts:
        for x in ts[t]['manual']:
            for y in ts[t]['automatic']:
                plt.figure()
                plt.plot(df[x],df[y],'o')
                plt.title(t)
                plt.ylabel(y)
                plt.xlabel(x)
                add_identity_lines()
                plt.show()

def plot_manual_x_vs_manual_y(df):
    ts = get_biomarker_tokens(df)
    for t in ts:
        # can do the following since get_biomarker_tokens guarantees that we have manual
        # fields
        xy = ts[t]["manual"]
        xy.sort()
        x = xy[0]
        y = xy[1]
        plt.figure()
        plt.title(t)
        plt.plot(df[x],df[y],'ko')
        plt.show()


def plot_auto_3D_vs_auto(df):
    ts = get_biomarker_tokens(df)
    for t in ts:
        # can do the following since get_biomarker_tokens guarantees that we have manual
        # fields
        xy = ts[t]["automatic"]
        xy.sort()
        x = xy[0]
        y = xy[2]
        plt.figure()
        plt.title(t)
        plt.plot(df[x],df[y],'ro')
        plt.xlabel(x)
        plt.ylabel(y)
        add_identity_line()
        plt.show()

def plot_auto_2D_vs_auto(df):
    ts = get_biomarker_tokens(df)
    for t in ts:
        # can do the following since get_biomarker_tokens guarantees that we have manual
        # fields
        xy = ts[t]["automatic"]
        xy.sort()
        x = xy[0]
        y = xy[1]
        plt.figure()
        plt.title(t)
        plt.plot(df[x],df[y],'go')
        plt.show()


def select_data_with_matching_gs(df, filename = "data.csv"):
    """
     Return dataframe with rows that have manual GS and for which GS is identical in both files
    :return:
    """
    #Remove rows with non-identical GS
    ts = get_biomarker_tokens(df)
    badrows = None
    for t in ts:
        x = "_".join((t,"manual","x"))
        y = "_".join((t,"manual","y"))
        current_bad_rows = df[x]!=df[y]
        if badrows is None:
            badrows = current_bad_rows
        else:
            badrows&=current_bad_rows
    d = df[~badrows] # Tilda for negation in pandas series, hello MATLAB
    return d

def save_tokens_as_csv(df,tokens,header = True):
    ts = get_biomarker_tokens(df)
    for t in tokens:
        if t not in ts:
            print("save_tokens_as_csv:token {} is not in the dataframe, ignored", t)
            continue
        tosave = sorted(ts[t]["automatic"])+[t+"_manual_x"]
        df.to_csv(t+".csv",columns=tosave,index=False,header=header)


if __name__=="__main__":
    try:
        fn = sys.argv[1:3]
    except:
        print("Usage: python look_at_the_data.py file1.xlsx file2.xlsx")
    d = get_records_with_manual(fn,True)
    d = select_data_with_matching_gs(d)
    # plot_auto_2D_vs_auto(d)
    # plot_auto_3D_vs_auto(d)
    # plot_manual_x_vs_manual_y(d)
    plot_auto_vs_manual(d)
    tokens_to_save = ["LVEDV"]
    save_tokens_as_csv(d, tokens_to_save)
