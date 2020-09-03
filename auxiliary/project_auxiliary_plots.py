"""This module contains auxiliary functions for plots that are used in the main notebook."""

########################  Installing the necessary packages  ########################
import pandas as pd
import numpy as np
import math
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import econtools
import econtools.metrics as mt
import statsmodels.api as smp
import patsy

##################### Figure 1 ######################
def figure1_nocontrol(data1, cols):
    """ Creates a data set to plot figure 1, Panel B, D, F.
    
    Args:
       - data1 (pd.DataFrame): the original data set
       - cols (list): a list of column names ["agus", "bct", "bcg"]
    Returns:
       - df_fig1 (pd.DataFrame): a data set for plotting panels with no controls
       
    """
    
    # A subset of the df, based on the mean values of agus, bcg, bct by dzagr01
    df_fig1 = data1.groupby("dzagr01")["agus", "bcg", "bct"].mean()
    df_fig1.reset_index(level = 0, inplace = True)
    
    for column in cols:
        fig1_A1 = sm.ols(formula = "{} ~ dzagr01".format(column), 
                   data = df_fig1[(df_fig1["dzagr01"] < 0)  & (abs(df_fig1["dzagr01"]) < 0.2)]).fit()
        fig1_A2 = sm.ols(formula = "{} ~ dzagr01".format(column), 
                   data = df_fig1[(df_fig1["dzagr01"] > 0)  & (abs(df_fig1["dzagr01"]) < 0.2)]).fit()
        pred_A1 = fig1_A1.predict()
        pred_A2 = fig1_A2.predict()
        df_fig1.loc[(df_fig1["dzagr01"] < 0)  & (abs(df_fig1["dzagr01"]) < 0.2), 
                          "pred_{}1".format(column)] = pred_A1
        df_fig1.loc[(df_fig1["dzagr01"] > 0)  & (abs(df_fig1["dzagr01"]) < 0.2), 
                          "pred_{}2".format(column)] = pred_A2
    return df_fig1

def figure1_control(data1, cols):
    """ Creates a data set to plot figure 1, Panel B, D, F.
    
    Args:
       - data1 (pd.DataFrame): the original data set
       - cols (list): a list of column names ["agus", "bct", "bcg"]
    Returns:
       - df_fig1_contr (pd.DataFrame): a data set for plotting panels with controls
       
    """
    data1["uazY"] = data1["uazY"].astype("category")
    for column in cols:
        data_df = data1.loc[(data1["dzagr01"] != 0) & (abs(data1["dzagr01"]) < 0.2), [column, "uazY"]].dropna()
        data_df["constant"] = [1] * len(data_df.index)
        y,X = patsy.dmatrices("{}~constant".format(column), data = data_df, return_type='dataframe')
        ybar = y.mean()
        y = y -  y.groupby(data_df["uazY"]).transform('mean') + ybar
        Xbar = X.mean()
        X = X - X.groupby(data_df["uazY"]).transform('mean') + Xbar
        reg = smp.OLS(y,X).fit()
        y_hat = reg.predict() 
        y_hat.shape = (len(y_hat), 1)             
        residual = y - y_hat
        data1["{}_res".format(column)] = residual
    df_fig1_contr = data1.groupby("dzagr01")["{}_res".format(cols[0]), 
                                             "{}_res".format(cols[1]), 
                                             "{}_res".format(cols[2])].mean()
    df_fig1_contr.reset_index(level = 0, inplace = True)
    
    for column in cols:
        fig1_B1 = sm.ols(formula = "{}_res ~ dzagr01".format(column), 
                   data = df_fig1_contr[(df_fig1_contr["dzagr01"] < 0)  & (abs(df_fig1_contr["dzagr01"]) < 0.2)]).fit()
        fig1_B2 = sm.ols(formula = "{}_res ~ dzagr01".format(column), 
                   data = df_fig1_contr[(df_fig1_contr["dzagr01"] > 0)  & (abs(df_fig1_contr["dzagr01"]) < 0.2)]).fit()
        pred_B1 = fig1_B1.predict()
        pred_B2 = fig1_B2.predict()
        df_fig1_contr.loc[(df_fig1_contr["dzagr01"] < 0)  & (abs(df_fig1_contr["dzagr01"]) < 0.2), 
                          "pred_{}1".format(column)] = pred_B1
        df_fig1_contr.loc[(df_fig1_contr["dzagr01"] > 0)  & (abs(df_fig1_contr["dzagr01"]) < 0.2), 
                          "pred_{}2".format(column)] = pred_B2
    return df_fig1_contr


def figure1(data1, data2):
    fig1, axes  = plt.subplots(3, 2, figsize = (15, 15))
    fig1.suptitle("Figure 1. Between-School Cutoffs: All Towns")
    # Figure 1: Panel A
    axes[0, 0].plot(data1["dzagr01"], data1["pred_agus1"])
    axes[0, 0].plot(data1["dzagr01"], data1["pred_agus2"])
    axes[0, 0].scatter(data1[abs(data1["dzagr01"]) < 0.2]["dzagr01"], data1[abs(data1["dzagr01"]) < 0.2]["agus"])
    axes[0, 0].axvline(0, 0, 7.7)
    axes[0, 0].set_title("Panel A. Average transition score - no controls")
    axes[0, 0].set_xlabel("Score distance to cutoff")
    axes[0, 0].set_ylabel("School level score")
    # Figure 1: Panel B
    axes[0, 1].plot(data2["dzagr01"], data2["pred_agus1"])
    axes[0, 1].plot(data2["dzagr01"], data2["pred_agus2"])
    axes[0, 1].scatter(data2[abs(data2["dzagr01"]) < 0.2]["dzagr01"], data2[abs(data2["dzagr01"]) < 0.2]["agus_res"])
    axes[0, 1].axvline(0)
    axes[0, 1].set_title("Panel B. Average transition score - controls")
    axes[0, 1].set_xlabel("Score distance to cutoff")
    axes[0, 1].set_ylabel("School level score")
    # Figure 1: Panel C
    axes[1, 0].plot(data1["dzagr01"], data1["pred_bct1"])
    axes[1, 0].plot(data1["dzagr01"], data1["pred_bct2"])
    axes[1, 0].scatter(data1[abs(data1["dzagr01"]) < 0.2]["dzagr01"], data1[abs(data1["dzagr01"]) < 0.2]["bct"])
    axes[1, 0].axvline(0)
    axes[1, 0].set_title("Panel C. Baccalaureate taken - no controls")
    axes[1, 0].set_xlabel("Score distance to cutoff")
    axes[1, 0].set_ylabel("Taken")
    # Figure 1: Panel D
    axes[1, 1].plot(data2["dzagr01"], data2["pred_bct1"])
    axes[1, 1].plot(data2["dzagr01"], data2["pred_bct2"])
    axes[1, 1].scatter(data2[abs(data2["dzagr01"]) < 0.2]["dzagr01"], data2[abs(data2["dzagr01"]) < 0.2]["bct_res"])
    axes[1, 1].axvline(0)
    axes[1, 1].set_title("Panel D. Baccalaureate taken - controls")
    axes[1, 1].set_xlabel("Score distance to cutoff")
    axes[1, 1].set_ylabel("Taken")
    # Figure 1: Panel E
    axes[2, 0].plot(data1["dzagr01"], data1["pred_bcg1"])
    axes[2, 0].plot(data1["dzagr01"], data1["pred_bcg2"])
    axes[2, 0].scatter(data1[abs(data1["dzagr01"]) < 0.2]["dzagr01"], data1[abs(data1["dzagr01"]) < 0.2]["bcg"])
    axes[2, 0].axvline(0)
    axes[2, 0].set_title("Panel E. Baccalaureate grade - no controls")
    axes[2, 0].set_xlabel("Score distance to cutoff")
    axes[2, 0].set_ylabel("Grade")
    # Figure 1: Panel F
    axes[2, 1].plot(data2["dzagr01"], data2["pred_bcg1"])
    axes[2, 1].plot(data2["dzagr01"], data2["pred_bcg2"])
    axes[2, 1].scatter(data2[abs(data2["dzagr01"]) < 0.2]["dzagr01"], data2[abs(data2["dzagr01"]) < 0.2]["bcg_res"])
    axes[2, 1].axvline(0)
    axes[2, 1].set_title("Panel F. Baccalaureate grade - controls")
    axes[2, 1].set_xlabel("Score distance to cutoff")
    axes[2, 1].set_ylabel("Grade")
    
    return 

###################################    Figure 2    #####################################

def figure2_df1(data1, cols):
    """ Creates a data set to plot figure 2, Panels A and B.
    
    Args:
       - data1 (pd.DataFrame): the original data set
       - cols (list): a list of column names ["first", "second"]
    Returns:
       - df_fig2 (pd.DataFrame): a data set for plotting panels A and B
       
    """
    
    
    # Data frame preparation for the figure 2: Panel A and B
    df_fig2 = data1[data1["Y"] == 1]
    df_fig2 = df_fig2[(df_fig2["z"] == df_fig2["nusua"] - 1) & (df_fig2["nusua"] >= 3)]
    df_fig2["first"] = np.where(df_fig2["uskua"] == df_fig2["nusua"], 1, 0)
    df_fig2["second"] = np.where(df_fig2["uskua"] == df_fig2["nusua"] - 1, 1, 0)
    df_fig2 = df_fig2.groupby("dzgr01")["first", "second"].mean()
    df_fig2.reset_index(level = 0, inplace = True)
    
    
    for column in cols:
        fig2_A1 = sm.ols(formula = "{} ~ dzgr01".format(column), 
                   data = df_fig2[(df_fig2["dzgr01"] < 0)  & (abs(df_fig2["dzgr01"]) < 0.2)]).fit()
        fig2_A2 = sm.ols(formula = "{} ~ dzgr01".format(column), 
                       data = df_fig2[(df_fig2["dzgr01"] > 0)  & (abs(df_fig2["dzgr01"]) < 0.2)]).fit()
        pred_f2A1 = fig2_A1.predict()
        pred_f2A2 = fig2_A2.predict()
        df_fig2.loc[(df_fig2["dzgr01"] < 0)  & (abs(df_fig2["dzgr01"]) < 0.2), "pred_f2A1{}".format(column)] = pred_f2A1
        df_fig2.loc[(df_fig2["dzgr01"] > 0)  & (abs(df_fig2["dzgr01"]) < 0.2), "pred_f2A2{}".format(column)] = pred_f2A2

    return df_fig2

def figure2_df2(data1, cols):
    """ Creates a data set to plot figure 2, Panels C and D.
    
    Args:
       - data1 (pd.DataFrame): the original data set
       - cols (list): a list of column names ["first", "second"]
    Returns:
       - df_fig2_2 (pd.DataFrame): a data set for plotting panels C and D
       
    """
    # Data preparation for figure 2: Panel C and D
    df_fig2_2 = data1[(data1["z"] == 1) & (data1["nusua"] >= 3)]
    df_fig2_2.loc[:, "first"] = np.select([df_fig2_2["uskua"] == 2,
                                          df_fig2_2["uskua"] != 2], [1, 0])
    df_fig2_2.loc[:, "second"] = np.select([df_fig2_2["uskua"] == 1,
                                           df_fig2_2["uskua"] != 1], [1, 0])
    df_fig2_2 = df_fig2_2.groupby("dzgr01")["first", "second"].mean()
    df_fig2_2.reset_index(level = 0, inplace = True)
    
    for column in cols:
        
        fig2_C1 = sm.ols(formula = "{} ~ dzgr01".format(column), 
               data = df_fig2_2[(df_fig2_2["dzgr01"] < 0)  & (abs(df_fig2_2["dzgr01"]) < 0.2)]).fit()
        fig2_C2 = sm.ols(formula = "{} ~ dzgr01".format(column), 
                       data = df_fig2_2[(df_fig2_2["dzgr01"] > 0)  & (abs(df_fig2_2["dzgr01"]) < 0.2)]).fit()
        pred_f2C1 = fig2_C1.predict()
        pred_f2C2 = fig2_C2.predict()
        df_fig2_2.loc[(df_fig2_2["dzgr01"] < 0)  & (abs(df_fig2_2["dzgr01"]) < 0.2), 
                      "pred_f2C1{}".format(column)] = pred_f2C1
        df_fig2_2.loc[(df_fig2_2["dzgr01"] > 0)  & (abs(df_fig2_2["dzgr01"]) < 0.2), 
                      "pred_f2C2{}".format(column)] = pred_f2C2
    return df_fig2_2

def figure2_df3(data1, cols):
    """ Creates a data set to plot figure 2, Panels E and F.
    
    Args:
       - data1 (pd.DataFrame): the original data set
       - cols (list): a list of column names ["first", "second"]
    Returns:
       - df_fig2_3 (pd.DataFrame): a data set for plotting panels E and F
       
    """
    
    # Data preparation for figure 2: Panel E and F
    df_fig2_3 = data1[data1["nusua"] == 2]
    df_fig2_3.loc[:, "first"] = np.where(df_fig2_3["uskua"] == df_fig2_3["nusua"], 1, 0)
    df_fig2_3.loc[:, "second"] = np.where(df_fig2_3["uskua"] == df_fig2_3["nusua"] - 1, 1, 0)
    df_fig2_3 = df_fig2_3.groupby("dzgr01")["first", "second"].mean()
    df_fig2_3.reset_index(level = 0, inplace = True)
    
    for column in cols:
        fig2_E1 = sm.ols(formula = "{} ~ dzgr01".format(column), 
               data = df_fig2_3[(df_fig2_3["dzgr01"] < 0)  & (abs(df_fig2_3["dzgr01"]) < 0.2)]).fit()
        fig2_E2 = sm.ols(formula = "{} ~ dzgr01".format(column), 
                       data = df_fig2_3[(df_fig2_3["dzgr01"] > 0)  & (abs(df_fig2_3["dzgr01"]) < 0.2)]).fit()
        pred_f2E1 = fig2_E1.predict()
        pred_f2E2 = fig2_E2.predict()
        df_fig2_3.loc[(df_fig2_3["dzgr01"] < 0)  & (abs(df_fig2_3["dzgr01"]) < 0.2), 
                      "pred_f2E1{}".format(column)] = pred_f2E1
        df_fig2_3.loc[(df_fig2_3["dzgr01"] > 0)  & (abs(df_fig2_3["dzgr01"]) < 0.2),
                      "pred_f2E2{}".format(column)] = pred_f2E2
    return df_fig2_3
    
# Bringing all together in one plot
def figure2(data1, data2, data3):
    fig2, axes  = plt.subplots(3, 2, figsize = (15, 15))
    fig2.suptitle("Figure 2. Top and Bottom Cutoffs in Towns with Three or More Schools: Two-School Towns")
    # Figure 2: Panel A
    axes[0, 0].plot(data1["dzgr01"], data1["pred_f2A1first"])
    axes[0, 0].plot(data1["dzgr01"], data1["pred_f2A2first"])
    axes[0, 0].scatter(data1[abs(data1["dzgr01"]) < 0.2]["dzgr01"], data1[abs(data1["dzgr01"]) < 0.2]["first"])
    axes[0, 0].axvline(0)
    axes[0, 0].set_title("Panel A. Cutoffs for the best two schools")
    axes[0, 0].set_xlabel("Score distance to cutoff")
    axes[0, 0].set_ylabel("Percent of students in school above cutoff")
    
    # Figure 2: Panel B
    axes[0, 1].plot(data1["dzgr01"], data1["pred_f2A1second"])
    axes[0, 1].plot(data1["dzgr01"], data1["pred_f2A2second"])
    axes[0, 1].scatter(data1[abs(data1["dzgr01"]) < 0.2]["dzgr01"], data1[abs(data1["dzgr01"]) < 0.2]["second"])
    axes[0, 1].axvline(0)
    axes[0, 1].set_title("Panel B. Cutoffs for the best two schools")
    axes[0, 1].set_xlabel("Score distance to cutoff")
    axes[0, 1].set_ylabel("Percent of students in school below cutoff")
    
    # Figure 2: Panel C
    axes[1, 0].plot(data2["dzgr01"], data2["pred_f2C1first"])
    axes[1, 0].plot(data2["dzgr01"], data2["pred_f2C2first"])
    axes[1, 0].scatter(data2[abs(data2["dzgr01"]) < 0.2]["dzgr01"], data2[abs(data2["dzgr01"]) < 0.2]["first"])
    axes[1, 0].axvline(0)
    axes[1, 0].set_title("Panel C. Cutoffs between the worst two schools")
    axes[1, 0].set_xlabel("Score distance to cutoff")
    axes[1, 0].set_ylabel("Percent of students in school above cutoff")
    
    # Figure 2: Panel D
    axes[1, 1].plot(data2["dzgr01"], data2["pred_f2C1second"])
    axes[1, 1].plot(data2["dzgr01"], data2["pred_f2C2second"])
    axes[1, 1].scatter(data2[abs(data2["dzgr01"]) < 0.2]["dzgr01"], data2[abs(data2["dzgr01"]) < 0.2]["second"])
    axes[1, 1].axvline(0)
    axes[1, 1].set_title("Panel D. Cutoffs between the worst two schools")
    axes[1, 1].set_xlabel("Score distance to cutoff")
    axes[1, 1].set_ylabel("Percent of students in school below cutoff")
    
    # Figure 2: Panel E
    axes[2, 0].plot(data3["dzgr01"], data3["pred_f2E1first"])
    axes[2, 0].plot(data3["dzgr01"], data3["pred_f2E2first"])
    axes[2, 0].scatter(data3[abs(data3["dzgr01"]) < 0.2]["dzgr01"], data3[abs(data3["dzgr01"]) < 0.2]["first"])
    axes[2, 0].axvline(0)
    axes[2, 0].set_title("Panel E. Cutoffs in two-school towns")
    axes[2, 0].set_xlabel("Score distance to cutoff")
    axes[2, 0].set_ylabel("Percent of students in school above cutoff")
    
    # Figure 2: Panel F
    axes[2, 1].plot(data3["dzgr01"], data3["pred_f2E1second"])
    axes[2, 1].plot(data3["dzgr01"], data3["pred_f2E2second"])
    axes[2, 1].scatter(data3[abs(data3["dzgr01"]) < 0.2]["dzgr01"], data3[abs(data3["dzgr01"]) < 0.2]["second"])
    axes[2, 1].axvline(0)
    axes[2, 1].set_title("Panel F. Cutoffs in two-school towns")
    axes[2, 1].set_xlabel("Score distance to cutoff")
    axes[2, 1].set_ylabel("Percent of students in school below cutoff")
    
    return 

################################       Figure 3      ###############################

def figure3_df1(data1, cols):
    """ Creates a data set to plot figure 3, Panels A and E.
    
    Args:
       - data1 (pd.DataFrame): the original data set
       - cols (list): a list of column names ["agus", "bcg"]
    Returns:
       - df_fig3 (pd.DataFrame): a data set for plotting panels A and E
       
    """
    
    # Data Preparation for figure 3: Panel A and E
    df_fig3 = data1[data1["zga"] > 7.74]
    df_fig3 = df_fig3.groupby("dzagr01")["agus", "bcg", "bct"].mean()
    df_fig3.reset_index(level = 0, inplace = True)
    
    for column in cols:
        fig3_A1 = sm.ols(formula = "{} ~ dzagr01".format(column), 
               data = df_fig3[(df_fig3["dzagr01"] < 0)  & (abs(df_fig3["dzagr01"]) < 0.2)]).fit()
        fig3_A2 = sm.ols(formula = "{} ~ dzagr01".format(column), 
                       data = df_fig3[(df_fig3["dzagr01"] > 0)  & (abs(df_fig3["dzagr01"]) < 0.2)]).fit()
        pred_f3A1 = fig3_A1.predict()
        pred_f3A2 = fig3_A2.predict()
        df_fig3.loc[(df_fig3["dzagr01"] < 0)  & (abs(df_fig3["dzagr01"]) < 0.2), 
                    "pred_f3A1{}".format(column)] = pred_f3A1
        df_fig3.loc[(df_fig3["dzagr01"] > 0)  & (abs(df_fig3["dzagr01"]) < 0.2), 
                    "pred_f3A2{}".format(column)] = pred_f3A2

    
    return df_fig3

def figure3_df2(data1, cols):
    """ Creates a data set to plot figure 3, Panels B and F.
    
    Args:
       - data1 (pd.DataFrame): the original data set
       - cols (list): a list of column names ["agus", "bcg"]
    Returns:
       - df_fig3_2 (pd.DataFrame): a data set for plotting panels B and F
       
    """
    
    # Data Preparation for figure 3: Panel B and F
    df_fig3_2 = data1[data1["zga"] > 7.74]
    
    for column in cols:
        fig3_dfb = df_fig3_2.loc[(df_fig3_2["dzagr01"] != 0) & (abs(df_fig3_2["dzagr01"]) < 0.2),
                                 ["{}".format(column), "uazY"]].dropna()
        fig3_dfb.loc[:, "constant"] = [1] * len(fig3_dfb.index)
        fig3_dfb.loc[:, "uazY"] = fig3_dfb["uazY"].astype("category")
        yb,Xb = patsy.dmatrices("{}~constant".format(column), fig3_dfb, return_type='dataframe')
        ybarb = yb.mean()
        yb = yb -  yb.groupby(fig3_dfb["uazY"]).transform('mean') + ybarb
        Xbarb = Xb.mean()
        Xb = Xb - Xb.groupby(fig3_dfb["uazY"]).transform('mean') + Xbarb
        reg = smp.OLS(yb,Xb).fit()
        y_hatb = reg.predict() 
        y_hatb.shape = (len(y_hatb), 1)             
        residual = yb - y_hatb
        df_fig3_2["{}_res".format(column)] = residual
    
    df_fig3_2 = df_fig3_2.groupby("dzagr01")["agus_res", "bcg_res"].mean()
    df_fig3_2.reset_index(level = 0, inplace = True)
    
    for column in cols:
        fig3_B1 = sm.ols(formula = "{}_res ~ dzagr01".format(column), 
               data = df_fig3_2[(df_fig3_2["dzagr01"] < 0)  & (abs(df_fig3_2["dzagr01"]) < 0.2)].fillna(0)).fit()
        fig3_B2 = sm.ols(formula = "{}_res ~ dzagr01".format(column), 
                       data = df_fig3_2[(df_fig3_2["dzagr01"] > 0)  & (abs(df_fig3_2["dzagr01"]) < 0.2)].fillna(0)).fit()
        pred_f3B1 = fig3_B1.predict()
        pred_f3B2 = fig3_B2.predict()
        df_fig3_2.loc[(df_fig3_2["dzagr01"] < 0)  & (abs(df_fig3_2["dzagr01"]) < 0.2),
                      "pred_f3B1{}".format(column)] = pred_f3B1
        df_fig3_2.loc[(df_fig3_2["dzagr01"] > 0)  & (abs(df_fig3_2["dzagr01"]) < 0.2), 
                      "pred_f3B2{}".format(column)] = pred_f3B2
        
    return df_fig3_2

def figure3_df3(data1, cols):
    """ Creates a data set to plot figure 3, Panels C and G.
    
    Args:
       - data1 (pd.DataFrame): the original data set
       - cols (list): a list of column names ["agus", "bcg"]
    Returns:
       - df_fig3_3 (pd.DataFrame): a data set for plotting panels C and G
       
    """
    
    
    # Data Preparation for figure 3: Panel C and G
    df_fig3_3 = data1[data1["zga"] < 6.76]
    df_fig3_3 = df_fig3_3.groupby("dzagr01")["agus", "bcg", "bct"].mean()
    df_fig3_3.reset_index(level = 0, inplace = True)
    
    for column in cols:
        fig3_C1 = sm.ols(formula = "{} ~ dzagr01".format(column), 
               data = df_fig3_3[(df_fig3_3["dzagr01"] < 0)  & (abs(df_fig3_3["dzagr01"]) < 0.2)]).fit()
        fig3_C2 = sm.ols(formula = "{} ~ dzagr01".format(column), 
                       data = df_fig3_3[(df_fig3_3["dzagr01"] > 0)  & (abs(df_fig3_3["dzagr01"]) < 0.2)]).fit()
        pred_f3C1 = fig3_C1.predict()

        pred_f3C2 = fig3_C2.predict()
        df_fig3_3.loc[(df_fig3_3["dzagr01"] < 0)  & (abs(df_fig3_3["dzagr01"]) < 0.2), 
                      "pred_f3C1{}".format(column)] = pred_f3C1
        df_fig3_3.loc[(df_fig3_3["dzagr01"] > 0)  & (abs(df_fig3_3["dzagr01"]) < 0.2), 
                      "pred_f3C2{}".format(column)] = pred_f3C2
    return df_fig3_3


def figure3_df4(data1, cols):
    """ Creates a data set to plot figure 3, Panels D and H.
    
    Args:
       - data1 (pd.DataFrame): the original data set
       - cols (list): a list of column names ["agus", "bcg"]
    Returns:
       - df_fig3_4 (pd.DataFrame): a data set for plotting panels D and H
       
    """
    
    
    # Data Preparation for figure3: Panel D and H
    df_fig3_4 = data1[data1["zga"] < 6.76]
    
    for column in cols:
        fig3_dfC = df_fig3_4.loc[(df_fig3_4["dzagr01"] != 0) & (abs(df_fig3_4["dzagr01"]) < 0.2),
                                 ["{}".format(column), "uazY"]].dropna()
        fig3_dfC.loc[:, "constant"] = [1] * len(fig3_dfC.index)
        fig3_dfC.loc[:, "uazY"] = fig3_dfC["uazY"].astype("category")
        yc,Xc = patsy.dmatrices("{} ~constant".format(column), fig3_dfC, return_type='dataframe')
        ybarc = yc.mean()
        yc = yc -  yc.groupby(fig3_dfC["uazY"]).transform('mean') + ybarc
        Xbarc = Xc.mean()
        Xc = Xc - Xc.groupby(fig3_dfC["uazY"]).transform('mean') + Xbarc
        reg = smp.OLS(yc,Xc).fit()
        y_hatc = reg.predict() 
        y_hatc.shape = (len(y_hatc), 1)             
        residual = yc - y_hatc
        df_fig3_4["{}_res".format(column)] = residual
    
    df_fig3_4 = df_fig3_4.groupby("dzagr01")["agus_res", "bcg_res"].mean()
    df_fig3_4.reset_index(level = 0, inplace = True)
    
    for column in cols:
        fig3_D1 = sm.ols(formula = "{}_res ~ dzagr01".format(column), 
               data = df_fig3_4[(df_fig3_4["dzagr01"] < 0)  & (abs(df_fig3_4["dzagr01"]) < 0.2)]).fit()
        fig3_D2 = sm.ols(formula = "{}_res ~ dzagr01".format(column), 
                       data = df_fig3_4[(df_fig3_4["dzagr01"] > 0)  & (abs(df_fig3_4["dzagr01"]) < 0.2)]).fit()
        pred_f3D1 = fig3_D1.predict()
        pred_f3D2 = fig3_D2.predict()
        df_fig3_4.loc[(df_fig3_4["dzagr01"] < 0)  & (abs(df_fig3_4["dzagr01"]) < 0.2),
                      "pred_f3D1{}".format(column)] = pred_f3D1
        df_fig3_4.loc[(df_fig3_4["dzagr01"] > 0)  & (abs(df_fig3_4["dzagr01"]) < 0.2), 
                      "pred_f3D2{}".format(column)] = pred_f3D2
        
    return df_fig3_4

# Bringing all together in one plot
def figure3(data1, data2, data3, data4):
    
    fig3, axes  = plt.subplots(4, 2, figsize = (17, 17))
    fig3.suptitle("Figure 3. Top and Bottom Terciles of Between-School Cutoffs")
    
    # Figure 3: Panel A
    axes[0, 0].plot(data1["dzagr01"], data1["pred_f3A1agus"])
    axes[0, 0].plot(data1["dzagr01"], data1["pred_f3A2agus"])
    axes[0, 0].scatter(data1[abs(data1["dzagr01"]) < 0.2]["dzagr01"], data1[abs(data1["dzagr01"]) < 0.2]["agus"])
    axes[0, 0].axvline(0)
    axes[0, 0].set_title("Panel A. Top cutoffs: avg. trans. score -- no controls")
    axes[0, 0].set_xlabel("Score distance to cutoff")
    axes[0, 0].set_ylabel("School level score")
    
    
    # Figure 3: Panel B
    axes[0, 1].plot(data2["dzagr01"], data2["pred_f3B1agus"])
    axes[0, 1].plot(data2["dzagr01"], data2["pred_f3B2agus"])
    axes[0, 1].scatter(data2[abs(data2["dzagr01"]) < 0.2]["dzagr01"], data2[abs(data2["dzagr01"]) < 0.2]["agus_res"])
    axes[0, 1].axvline(0)
    axes[0, 1].set_title("Panel B: Top cutoffs--Avg. trans. score--controls")
    axes[0, 1].set_xlabel("Score distance to cutoff")
    axes[0, 1].set_ylabel("School level score")
    
    
    # Figure 3: Panel C
    axes[1, 0].plot(data3["dzagr01"], data3["pred_f3C1agus"])
    axes[1, 0].plot(data3["dzagr01"], data3["pred_f3C2agus"])
    axes[1, 0].scatter(data3[abs(data3["dzagr01"]) < 0.2]["dzagr01"], data3[abs(data3["dzagr01"]) < 0.2]["agus"])
    axes[1, 0].axvline(0)
    axes[1, 0].set_title("Panel C: Bottom cutoffs--Avg. trans. score--no controls")
    axes[1, 0].set_xlabel("Score distance to cutoff")
    axes[1, 0].set_ylabel("School Level Score")
    
    
    # Figure 3: Panel D
    axes[1, 1].plot(data4["dzagr01"], data4["pred_f3D1agus"])
    axes[1, 1].plot(data4["dzagr01"], data4["pred_f3D2agus"])
    axes[1, 1].scatter(data4[abs(data4["dzagr01"]) < 0.2]["dzagr01"], data4[abs(data4["dzagr01"]) < 0.2]["agus_res"])
    axes[1, 1].axvline(0)
    axes[1, 1].set_title("Panel D: Bottom cutoffs--Avg. transition score--controls")
    axes[1, 1].set_xlabel("Score distance to cutoff")
    axes[1, 1].set_ylabel("School Level Score")
    
    
    # Figure 3: Panel E
    axes[2, 0].plot(data1["dzagr01"], data1["pred_f3A1bcg"])
    axes[2, 0].plot(data1["dzagr01"], data1["pred_f3A2bcg"])
    axes[2, 0].scatter(data1[abs(data1["dzagr01"]) < 0.2]["dzagr01"], data1[abs(data1["dzagr01"]) < 0.2]["bcg"])
    axes[2, 0].axvline(0)
    axes[2, 0].set_title("Panel E. Top cutoffs: Bacc. grade -- no controls")
    axes[2, 0].set_xlabel("Score distance to cutoff")
    axes[2, 0].set_ylabel("Grade")
    
    
    # Figure 3: Panel F
    axes[2, 1].plot(data2["dzagr01"], data2["pred_f3B1bcg"])
    axes[2, 1].plot(data2["dzagr01"], data2["pred_f3B2bcg"])
    axes[2, 1].scatter(data2[abs(data2["dzagr01"]) < 0.2]["dzagr01"], data2[abs(data2["dzagr01"]) < 0.2]["bcg_res"])
    axes[2, 1].axvline(0)
    axes[2, 1].set_title("Panel F: Top cutoffs--Bacc. grade--controls")
    axes[2, 1].set_xlabel("Score distance to cutoff")
    axes[2, 1].set_ylabel("Grade")
    
    
    # Figure 3: Panel G 
    axes[3, 0].plot(data3["dzagr01"], data3["pred_f3C1bcg"])
    axes[3, 0].plot(data3["dzagr01"], data3["pred_f3C2bcg"])
    axes[3, 0].scatter(data3[abs(data3["dzagr01"]) < 0.2]["dzagr01"], data3[abs(data3["dzagr01"]) < 0.2]["bcg"])
    axes[3, 0].axvline(0)
    axes[3, 0].set_title("Panel G: Bottom cutoffs--Bacc. grade--no controls")
    axes[3, 0].set_xlabel("Score distance to cutoff")
    axes[3, 0].set_ylabel("Grade")
    
    
    # Figure 3: Panel H
    axes[3, 1].plot(data4["dzagr01"], data4["pred_f3D1bcg"])
    axes[3, 1].plot(data4["dzagr01"], data4["pred_f3D2bcg"])
    axes[3, 1].scatter(data4[abs(data4["dzagr01"]) < 0.2]["dzagr01"], data4[abs(data4["dzagr01"]) < 0.2]["bcg_res"])
    axes[3, 1].axvline(0)
    axes[3, 1].set_title("Panel H: Bottom cutoffs--Bacc. grade--controls")
    axes[3, 1].set_xlabel("Score distance to cutoff")
    axes[3, 1].set_ylabel("Grade")
    plt.tight_layout(pad=6, w_pad=2, h_pad=3)
    
    
    return 


    

    