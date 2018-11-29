import sys
import pandas as pd


if __name__=="__main__":
    try:
        fn = sys.argv[1]
    except:
        print("Usage: python custom_preprocessing.py file1.xlsx")
    dfr = pd.read_excel(fn)
    # datify:
    truth = ['leftch4p']
    data = [
        'lhCortexVol',
        'lhCerebralWhiteMatterVol',
        'left_Hippocampal_tail',
        'left_subiculum',
        'left_CA1',
        'left_hippocampal_fissure',
        'left_presubiculum',
        'left_parasubiculum',
        'left_molecular_layer_HP',
        'left_GC_ML_DG',
        'left_CA3',
        'left_CA4',
        'left_fimbria',
        'left_HATA',
        'left_Whole_hippocampus',
    ]
    tosave = truth+data
    # [cn in df for cn in tosave]
    ## normalize by TIV
    df = dfr.reindex(columns=tosave).div(dfr.TIV,axis='index')
    ## Save
    df.to_csv('_data.csv',columns=data, index=False, header=False)
    df.to_csv('_truth.csv',columns=truth, index=False, header=False)

    # trut

