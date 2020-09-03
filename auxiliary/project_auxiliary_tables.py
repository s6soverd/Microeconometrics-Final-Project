"""This module contains auxiliary functions for the creation of tables in the main notebook."""

from auxiliary.project_auxiliary_plots import *
from auxiliary.project_auxiliary_tables import *

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



########################    Table1   ########################
def table1(data1, data2, data3):
    
    """ Creates Table 1-descriptive statistics of 2001-2003 cohorts """
    
    # Panel A: All towns
    # Panel A.1. Individual level
  
    panel_A1_1 = pd.DataFrame()
    list_df = [data1, data2, data3]
    for data_frame in list_df:
        panel_A_df = data_frame[["grade","bct", "bcg"]].describe().iloc[0:3,].T
        panel_A1_1 = pd.concat([panel_A1_1, panel_A_df], axis = 1)
    panel_A1_1.rename(index = {"grade": "Transition grade",
                               "bct":   "Baccalaureate taken",
                               "bcg":   "Baccalaureate grade"}, inplace = True)
    panel_A1_1.index.name = "Panel A.1. Individual Level"
    
    # Panel A.2. Track level
    
    panel_A2_1 = pd.DataFrame()
    list_df = [data1, data2, data3]
    for data_frame in list_df:
        panel_A_df = data_frame.groupby("us2").agg({"ct": "sum",
                        "us": "mean",
                        "ua": "mean"})
        panel_A_df.reset_index(level = 0, inplace = True)
        panel_A2_1 = pd.concat([panel_A2_1, panel_A_df["ct"].describe().iloc[0:3]], axis = 0, sort = True)
    panel_A2_1 = panel_A2_1.T.rename(index = {0: "Number of ninth grade students"})
    panel_A2_1.index.name = "Panel A.2. Track Level"
   
    # Panel A.3. School level
    
    panel_A3_1 = pd.DataFrame()
    for data_frame in list_df:
        panel_A_df = data_frame.groupby("us2").agg({"ct": "sum",
                        "us": "mean",
                        "ua": "mean"})
        panel_A_df.reset_index(level = 0, inplace = True)
        panel_A_df.loc[:, "ct"] = 1
        panel_A3_1 = pd.concat([panel_A3_1, panel_A_df.groupby("us")["ct"].sum().describe().iloc[0:3,]], axis = 0, sort = True)
    panel_A3_2 = pd.DataFrame()
    for data_frame in list_df:
        panel_A_df = data_frame.groupby("us").agg({"ct": "sum",
                                         "ua": "mean"})
        panel_A_df.reset_index(level = 0, inplace = True)
        panel_A3_2 = pd.concat([panel_A3_2, panel_A_df["ct"].describe().iloc[0:3, ].T], axis = 0, sort = True)
    panel_A3 = panel_A3_2.T.append(panel_A3_1.T, ignore_index = True).rename(index = {0: "Number of ninth grade students", 
                                                                                      1: "Number of Tracks"})
    panel_A3.index.name = "Panel A.3. School Level"
   
    # Panel A.4. Town level
    
    panel_A4_1 = pd.DataFrame()
    for data_frame in list_df:
        panel_A_df = data_frame.groupby("us2").agg({"ct": "sum",
                        "us": "mean",
                        "ua": "mean"})
        panel_A_df.reset_index(level = 0, inplace = True)
        panel_A_df.loc[:, "ct"] = 1
        panel_A4_1 = pd.concat([panel_A4_1, panel_A_df.groupby("ua")["ct"].sum().describe().iloc[0:3, ]], axis = 0, sort = True)
    panel_A4_2 = pd.DataFrame()
    for data_frame in list_df:
        panel_A_df = data_frame.groupby("us").agg({"ct": "sum",
                        "ua": "mean"})
        panel_A_df.reset_index(level = 0, inplace = True)
        panel_A_df.loc[:, "ct"] = 1
        panel_A4_2 = pd.concat([panel_A4_2, panel_A_df.groupby("ua")["ct"].sum().describe().iloc[0:3, ]], axis = 0, sort = True)
    panel_A4_3 = pd.DataFrame()
    for data_frame in list_df:
        panel_A4_3 = pd.concat([panel_A4_3, data_frame.groupby("ua")["ct"].sum().describe().iloc[0:3,]], axis = 0, sort = True)
        
    panel_A4 = panel_A4_3.T.append([panel_A4_2.T, panel_A4_1.T], 
                               ignore_index = True).rename(index = {0: "Number of ninth grade students",
                                                                    1: "Number of Schools",
                                                                    2: "Number of tracks"})
    panel_A4.index.name = "Panel A.4. Town Level"
   

    # Panel B. Survey towns
    # Panel B.1. Individual Level 
    
    panel_B1_1 = pd.DataFrame()
    list_df = [data1, data2, data3]
    for data_frame in list_df:
        panel_B_df = data_frame[data_frame["survey"] == 1][["grade","bct", "bcg"]].describe().iloc[0:3,].T
        panel_B1_1 = pd.concat([panel_B1_1, panel_B_df], axis = 1)
    panel_B1_1.rename(index = {"grade": "Transition grade",
                               "bct":   "Baccalaureate taken",
                               "bcg":   "Baccalaureate grade"}, inplace = True)
    panel_B1_1.index.name = "Panel B.1. Individual Level"
    
    # Panel B.2. Track Level
    
    panel_B2_1 = pd.DataFrame()
    list_df = [data1, data2, data3]
    for data_frame in list_df:
        panel_B_df = data_frame[data_frame["survey"] == 1].groupby("us2").agg({"ct": "sum",
                        "us": "mean",
                        "ua": "mean"})
        panel_B_df.reset_index(level = 0, inplace = True)
        panel_B2_1 = pd.concat([panel_B2_1, panel_B_df["ct"].describe().iloc[0:3]], axis = 0, sort = True)
    panel_B2_1 = panel_B2_1.T.rename(index = {0: "Number of ninth grade students"})
    panel_B2_1.index.name = "Panel B.2. Track Level"
   
    # Panel B.3. School Level
    
    panel_B3_1 = pd.DataFrame()
    for data_frame in list_df:
        panel_B_df = data_frame[data_frame["survey"] == 1].groupby("us2").agg({"ct": "sum",
                        "us": "mean",
                        "ua": "mean"})
        panel_B_df.reset_index(level = 0, inplace = True)
        panel_B_df.loc[:, "ct"] = 1
        panel_B3_1 = pd.concat([panel_B3_1, panel_B_df.groupby("us")["ct"].sum().describe().iloc[0:3,]], axis = 0, sort = True)
    panel_B3_2 = pd.DataFrame()
    for data_frame in list_df:
        panel_B_df = data_frame[data_frame["survey"] == 1].groupby("us").agg({"ct": "sum",
                                         "ua": "mean"})
        panel_B_df.reset_index(level = 0, inplace = True)
        panel_B3_2 = pd.concat([panel_B3_2, panel_B_df["ct"].describe().iloc[0:3, ].T], axis = 0, sort = True)
    panel_B3 = panel_B3_2.T.append(panel_B3_1.T, ignore_index = True).rename(index = {0: "Number of ninth grade students", 
                                                                                      1: "Number of Tracks"})
    panel_B3.index.name = "Panel B.3. School Level"
    
    # Panel B.4. Town Level
    
    panel_B4_1 = pd.DataFrame()
    for data_frame in list_df:
        panel_B_df = data_frame[data_frame["survey"] == 1].groupby("us2").agg({"ct": "sum",
                        "us": "mean",
                        "ua": "mean"})
        panel_B_df.reset_index(level = 0, inplace = True)
        panel_B_df.loc[:, "ct"] = 1
        panel_B4_1 = pd.concat([panel_B4_1, panel_B_df.groupby("ua")["ct"].sum().describe().iloc[0:3, ]], axis = 0, sort = True)
    panel_B4_2 = pd.DataFrame()
    for data_frame in list_df:
        panel_B_df = data_frame[data_frame["survey"] == 1].groupby("us").agg({"ct": "sum",
                        "ua": "mean"})
        panel_B_df.reset_index(level = 0, inplace = True)
        panel_B_df.loc[:, "ct"] = 1
        panel_B4_2 = pd.concat([panel_B4_2, panel_B_df.groupby("ua")["ct"].sum().describe().iloc[0:3, ]], axis = 0, sort = True)
    panel_B4_3 = pd.DataFrame()
    for data_frame in list_df:
        panel_B4_3 = pd.concat([panel_B4_3, data_frame[data_frame["survey"] == 1].groupby("ua")["ct"].sum().describe().iloc[0:3,]], 
                                                                                        axis = 0, sort = True)
        
    panel_B4 = panel_B4_3.T.append([panel_B4_2.T, panel_B4_1.T], 
                               ignore_index = True).rename(index = {0: "Number of ninth grade students",
                                                                    1: "Number of Schools",
                                                                    2: "Number of tracks"})
    panel_B4.index.name = "Panel B.4. Town Level"
    
    panels_list = [panel_A1_1, panel_A2_1, panel_A3, panel_A4, panel_B1_1, panel_B2_1, panel_B3, panel_B4]
    for panel in panels_list:
        panel.columns = ["Obs. 2001", "Mean 2001", "Sd 2001",  "Obs. 2002",  
                   "Mean 2002",  "Sd 2002",  "Obs. 2003", "Mean 2003", "Std 2003"]
        
    
    
             
    return [panel_A1_1, panel_A2_1, panel_A3, panel_A4, panel_B1_1, panel_B2_1, panel_B3, panel_B4]


####################################   Table2    ############################################


def table2(data1):
    
    """ Creates Table 2 - Descriptive Statistics for Survey Data - 2005-2007 cohorts """
    
    # Panel A. Socioeconomic characteristics (Household Survey)
    panel_A_1 = data1[["head_sex", "head_age"]].describe().iloc[0:3, ].T
    panel_A_1.index.name = "Socioeconomic Characteristics"
    panel_A_1.rename(index = {"head_sex": "Female head of household",
                              "head_age": "Age of household head"}, inplace = True)
    # Ethnicity of household head
    panel_A_2 = data1[["head_nat_romanian", "head_nat_hungarian", "head_nat_gypsy", "head_nat_other"]].describe().iloc[0:3, ].T
    panel_A_2.index.name = "Ethnicity of household head"
    panel_A_2.rename(index = {"head_nat_romanian": "Romanian", 
                              "head_nat_hungarian": "Hungarian", 
                              "head_nat_gypsy": "Gypsy",
                              "head_nat_other": "Other"}, inplace = True)
    # Education of household head
    panel_A_3 = data1[["head_educ_primary", "head_educ_sec", "head_educ_tertiary"]].describe().iloc[0:3, ].T
    panel_A_3.index.name = "Education of household head"
    panel_A_3.rename(index = {"head_educ_primary": "Primary",
                              "head_educ_sec": "Secondary",
                              "head_educ_tertiary": "Tertiary"}, inplace = True)
    # Child gender and age
    panel_A_4 = data1[["ch_sex", "ch_age"]].describe().iloc[0:3, ].T
    panel_A_4.index.name = "Age and Sex of the child"
    panel_A_4.rename(index = {"ch_sex": "Female child",
                              "ch_age": "Age of child"}, inplace = True)
    # Panel B. Parental Responses
    panel_B_1 = data1[["p_d_parent_volunteer", "p_d_homework_help", "p_d_homework"]].describe().iloc[0:3, ].T
    data1["p_tutoring"] = data1["p_tutoring"].astype("category")
    panel_B_1_categorical = data1["p_tutoring"].cat.codes.describe().iloc[1:3,].T
    categ = pd.DataFrame(panel_B_1_categorical).T
    categ.loc[:, "count"]  = data1["p_tutoring"].value_counts().sum()
    cols = categ.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    categ = categ[cols]

    panel_B1 = panel_B_1.append(categ).rename(index = {"p_d_parent_volunteer": "Parent has volunteered at school in the past 12 months",
                          0: "Parent has paid for tutoring services in the past 12 months",
                          "p_d_homework_help": "Parent helps child with homework every day or almost every day",
                           "p_d_homework":  "Child does homework every day or almost every day"})
    panel_B1.index.name = "Parental responses"
    
   # Panel C. Child Responses
    panel_C_1 = data1[["ch_peers_index_bad", "ch_d_homework", "ch_rank_homework_index"]].describe().iloc[0:3, ].T
    count = data1["ch_rank_peers"].value_counts().sum()
    data1["ch_rank_peers"] = data1["ch_rank_peers"].astype("category")
    data1["ch_rank_peers"] = data1["ch_rank_peers"].cat.rename_categories([7.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0])
    mean = (4 *data1["ch_rank_peers"].value_counts()[4]  + 5 *data1["ch_rank_peers"].value_counts()[5]  
            + 6 *data1["ch_rank_peers"].value_counts()[6]  + 3 *data1["ch_rank_peers"].value_counts()[3] 
            + data1["ch_rank_peers"].value_counts()[-3] + 2 * data1["ch_rank_peers"].value_counts()[2] + 
             7 * data1["ch_rank_peers"].value_counts()[-1]) * 1/ data1["ch_rank_peers"].value_counts().sum()
    mean_array = [mean] * len(data1.index)
    mean_series = pd.Series(mean_array)
    inside = (data1["ch_rank_peers"].subtract(mean_series)).apply(lambda x: x**2)
    std = math.sqrt(inside.sum() / len(data1.index))
    categ_C = pd.DataFrame({"count": count,
                      "mean": mean,
                      "std": std}, index = ["Relative rank among peers"])
    panel_C = panel_C_1.append(categ_C).rename(index = {"ch_peers_index_bad": "Index of negative interactions with peers",
                                        "ch_d_homework": "Child does homework daily or almost daily",
                                        "ch_rank_homework_index": "Child perceives homework to be easy"})
    panel_C.index.name = "Child responses"
    
    # Panel D. Language teacher qualifications
    
    panel_D = data1[["didactic_Romanian", "novice_Romanian"]].describe().iloc[0:3].T
    panel_D.index.name = "Language Teacher qualifications"
    panel_D.rename(index = {"didactic_Romanian": "Proportion of teachers with highest state certification",
                       "novice_Romanian": "Proportion of teachers who are novices"}, inplace = True)
    
    return [panel_A_1, panel_A_2, panel_A_3, panel_A_4, panel_B1, panel_C, panel_D]   


#######################################    Table3    #########################################

def areg(formula,data=None,absorb=None,cluster=None): 
    
    """ """

    y,X = patsy.dmatrices(formula,data,return_type='dataframe')

    ybar = y.mean()
    y = y -  y.groupby(data[absorb]).transform('mean') + ybar

    Xbar = X.mean()
    X = X - X.groupby(data[absorb]).transform('mean') + Xbar

    reg = smp.OLS(y,X)
    

    return reg.fit(cov_type='cluster',cov_kwds={'groups':data[cluster].values})

# Panel A

def table3_panelA(data1, ik_list):
    
    # Panel A. School level average transition grade: 2001-2003 cohorts - between school cutoffs
    # All towns
    regression1 = areg(formula = "agus ~ dga + dzag + dzag_after", 
         data = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < 1), ["agus","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression2 = areg(formula = "agus ~ dga + dzag + dzag_after", 
         data = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < ik_list[0]), ["agus","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    params1 = {"R squared": [regression1.rsquared, regression2.rsquared],
          "1{Transition grade >= cutoff}": [regression1.params[1], regression2.params[1]],
          "Standard Error": [regression1.bse[1],regression2.bse[1]],
          "P-value": [regression1.pvalues[1], regression2.pvalues[1]]}
    panel_A_1 = pd.DataFrame(data = params1, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_A_1.index.name = "Panel A.School-level average transition grade: 2001-2003, All towns, between-school cutoffs"
    # Survey towns
    regression3 = areg(formula = "agus ~ dga + dzag + dzag_after", 
         data = data1[data1["survey"] == 1].loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < 1),
                                ["agus","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression4 = areg(formula = "agus ~ dga + dzag + dzag_after", 
         data = data1[data1["survey"] == 1].loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < ik_list[1]),
                                ["agus","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    params2 = {"R squared": [regression3.rsquared, regression4.rsquared],
          "1{Transition grade >= cutoff}": [regression3.params[1], regression4.params[1]],
          "Standard Error": [regression3.bse[1],regression4.bse[1]],
          "P-value": [regression3.pvalues[1], regression4.pvalues[1]]}
    panel_A_2 = pd.DataFrame(data = params2, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_A_2.index.name = "Panel A.School-level average transition grade: 2001-2003, Survey towns, between-school cutoffs"
    
    return [panel_A_1, panel_A_2]

# Panel B

def table3_panelB(data3, ik_list):
    
    # Panel B. Track-level average transition grade: 2001-2003 cohorts -- between track cutoffs
    # All towns
    regression5 = areg(formula = "agus2B ~ dga + dzag + dzag_after", 
         data = data3.loc[(data3["dzag"] != 0) & (abs(data3["dzag"]) < 1), ["agus2B","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression6 = areg(formula = "agus2B ~ dga + dzag + dzag_after", 
         data = data3.loc[(data3["dzag"] != 0) & (abs(data3["dzag"]) < ik_list[2]), ["agus2B","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    params3 = {"R squared": [regression5.rsquared, regression6.rsquared],
          "1{Transition grade >= cutoff}": [regression5.params[1], regression6.params[1]],
          "Standard Error": [regression5.bse[1],regression6.bse[1]],
          "P-value": [regression5.pvalues[1], regression6.pvalues[1]]}
    panel_B_1 = pd.DataFrame(data = params3, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_B_1.index.name = "Panel B.Track-level average transition grade: 2001-2003, All towns, between-track cutoffs"
    # Survey towns
    regression7 = areg(formula = "agus2B ~ dga + dzag + dzag_after", 
         data = data3[data3["survey"] == 1].loc[(data3["dzag"] != 0) & (abs(data3["dzag"]) < 1),
                                ["agus2B","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression8 = areg(formula = "agus2B ~ dga + dzag + dzag_after", 
         data = data3[data3["survey"] == 1].loc[(data3["dzag"] != 0) & (abs(data3["dzag"]) < ik_list[3]),
                                ["agus2B","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    params4 = {"R squared": [regression7.rsquared, regression8.rsquared],
          "1{Transition grade >= cutoff}": [regression7.params[1], regression8.params[1]],
          "Standard Error": [regression7.bse[1],regression8.bse[1]],
          "P-value": [regression7.pvalues[1], regression8.pvalues[1]]}
    panel_B_2 = pd.DataFrame(data = params4, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_B_2.index.name = "Panel B.Track-level average transition grade: 2001-2003, Survey towns, between-track cutoffs"
    
    return [panel_B_1, panel_B_2]

# Panel C

def table3_panelC(data1, ik_list):
    # Panel C. Track level average transition grade grade: 2001-2003 -- between school cutoffs
    # All towns
    regression9 = areg(formula = "agus2 ~ dga + dzag + dzag_after", 
         data = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < 1), ["agus2","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression10 = areg(formula = "agus2 ~ dga + dzag + dzag_after", 
         data = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < ik_list[4]), ["agus2","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    params5 = {"R squared": [regression9.rsquared, regression10.rsquared],
          "1{Transition grade >= cutoff}": [regression9.params[1], regression10.params[1]],
          "Standard Error": [regression9.bse[1],regression10.bse[1]],
          "P-value": [regression9.pvalues[1], regression10.pvalues[1]]}
    panel_C_1 = pd.DataFrame(data = params5, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_C_1.index.name = "Panel C.Track-level average transition grade: 2001-2003, All towns, between-school cutoffs"
    # Survey towns
    regression11 = areg(formula = "agus2 ~ dga + dzag + dzag_after", 
         data = data1[data1["survey"] == 1].loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < 1),
                                ["agus2","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression12 = areg(formula = "agus2 ~ dga + dzag + dzag_after", 
         data = data1[data1["survey"] == 1].loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < ik_list[5]),
                                ["agus2","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    params6 = {"R squared": [regression11.rsquared, regression12.rsquared],
          "1{Transition grade >= cutoff}": [regression11.params[1], regression12.params[1]],
          "Standard Error": [regression11.bse[1],regression12.bse[1]],
          "P-value": [regression11.pvalues[1], regression12.pvalues[1]]}
    panel_C_2 = pd.DataFrame(data = params6, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_C_2.index.name = "Panel C.Track-level average transition grade: 2001-2003, Survey towns, betweem-school cutoffs"
    
    return [panel_C_1, panel_C_2]

# Panel D

def table3_panelD(data2, ik_list):
    # Panel D. School level transition grade: 2005-2007 cohorts -- between school cutoffs
    # All towns
    regression13 = areg(formula = "agus ~ dga + dzag + dzag_after", 
         data = data2.loc[(data2["dzag"] != 0) & (abs(data2["dzag"]) < 1), ["agus","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression14 = areg(formula = "agus ~ dga + dzag + dzag_after", 
         data = data2.loc[(data2["dzag"] != 0) & (abs(data2["dzag"]) < ik_list[6]), ["agus","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    params7 = {"R squared": [regression13.rsquared, regression14.rsquared],
          "1{Transition grade >= cutoff}": [regression13.params[1], regression14.params[1]],
          "Standard Error": [regression13.bse[1],regression14.bse[1]],
          "P-value": [regression13.pvalues[1], regression14.pvalues[1]]}
    panel_D_1 = pd.DataFrame(data = params7, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_D_1.index.name = "Panel D.School-level average transition grade: 2005-2007, All towns, between-school cutoffs"
    # Survey towns
    regression15 = areg(formula = "agus ~ dga + dzag + dzag_after", 
         data = data2[data2["survey"] == 1].loc[(data2["dzag"] != 0) & (abs(data2["dzag"]) < 1),
                                ["agus","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression16 = areg(formula = "agus ~ dga + dzag + dzag_after", 
         data = data2[data2["survey"] == 1].loc[(data2["dzag"] != 0) & (abs(data2["dzag"]) < ik_list[7]),
                                ["agus","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    params8 = {"R squared": [regression15.rsquared, regression16.rsquared],
          "1{Transition grade >= cutoff}": [regression15.params[1], regression16.params[1]],
          "Standard Error": [regression15.bse[1],regression16.bse[1]],
          "P-value": [regression15.pvalues[1], regression16.pvalues[1]]}
    panel_D_2 = pd.DataFrame(data = params8, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_D_2.index.name = "Panel D.School-level average transition grade: 2005-2007, Survey towns, between-school cutoffs"
    
    return [panel_D_1, panel_D_2]

###############################     Table 4      #################################
def demean(formula, data, absorb = None):
    
    
    y,X = patsy.dmatrices(formula,data,return_type='dataframe')

    ybar = y.mean()
    y = y -  y.groupby(data[absorb]).transform('mean') + ybar

    Xbar = X.mean()
    X = X - X.groupby(data[absorb]).transform('mean') + Xbar
    df = pd.concat([y, X], axis = 1)
    return df

def table4_panelA(data1, ik_list_table4):
    
    # Panel A. Baccalaureate taken dummy: 2001-2003 cohorts - between school cutoffs
    # All towns
    regression1 = areg(formula = "bct ~ dga + dzag + dzag_after", 
         data = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < 1), ["bct","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression2 = areg(formula = "bct ~ dga + dzag + dzag_after", 
         data = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < ik_list_table4[0]),["bct","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    params1 = {"R squared": [regression1.rsquared, regression2.rsquared],
          "1{Transition grade >= cutoff}": [regression1.params[1], regression2.params[1]],
          "Standard Error": [regression1.bse[1],regression2.bse[1]],
          "P-value": [regression1.pvalues[1], regression2.pvalues[1]]}
    panel_A_1 = pd.DataFrame(data = params1, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_A_1.index.name = "Panel A.Baccalaureate taken dummy: 2001-2003, All towns, between-school cutoffs"
    # Survey towns
    regression3 = areg(formula = "bct ~ dga + dzag + dzag_after", 
         data = data1[data1["survey"] == 1].loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < 1),
                                ["bct","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression4 = areg(formula = "bct ~ dga + dzag + dzag_after", 
         data = data1[data1["survey"] == 1].loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < ik_list_table4[1]),
                                ["bct","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    params2 = {"R squared": [regression3.rsquared, regression4.rsquared],
          "1{Transition grade >= cutoff}": [regression3.params[1], regression4.params[1]],
          "Standard Error": [regression3.bse[1],regression4.bse[1]],
          "P-value": [regression3.pvalues[1], regression4.pvalues[1]]}
    panel_A_2 = pd.DataFrame(data = params2, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_A_2.index.name = "Panel A.Baccalaureate taken dummy: 2001-2003, Survey towns, between-school cutoffs"
    
    return [panel_A_1, panel_A_2]

# Panel B

def table4_panelB(data1, ik_list_table4):
    # Panel B. Baccalaureate grade: 2001-2003 cohorts - between school cutoffs
    # All towns
    regression1 = areg(formula = "bcg ~ dga + dzag + dzag_after", 
         data = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < 1), ["bcg","dga","dzag","dzag_after","sid2","uazY"]].dropna(),
         absorb = "uazY", cluster = "sid2")
    regression2 = areg(formula = "bcg ~ dga + dzag + dzag_after", 
         data = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < ik_list_table4[2]), 
                          ["bcg","dga","dzag","dzag_after","sid2","uazY"]].dropna(),
         absorb = "uazY", cluster = "sid2")
    params1 = {"R squared": [regression1.rsquared, regression2.rsquared],
          "1{Transition grade >= cutoff}": [regression1.params[1], regression2.params[1]],
          "Standard Error": [regression1.bse[1],regression2.bse[1]],
          "P-value": [regression1.pvalues[1], regression2.pvalues[1]]}
    panel_B_1 = pd.DataFrame(data = params1, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_B_1.index.name = "Panel B.Baccalaureate grade: 2001-2003, All towns, between-school cutoffs"
    # Survey towns
    regression3 = areg(formula = "bcg ~ dga + dzag + dzag_after", 
         data = data1[data1["survey"] == 1].loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < 1),
                                ["bcg","dga","dzag","dzag_after","sid2","uazY"]].dropna(),
         absorb = "uazY", cluster = "sid2")
    regression4 = areg(formula = "bcg ~ dga + dzag + dzag_after", 
         data = data1[data1["survey"] == 1].loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < ik_list_table4[3]),
                                ["bcg","dga","dzag","dzag_after","sid2","uazY"]].dropna(),
         absorb = "uazY", cluster = "sid2")
    params2 = {"R squared": [regression3.rsquared, regression4.rsquared],
          "1{Transition grade >= cutoff}": [regression3.params[1], regression4.params[1]],
          "Standard Error": [regression3.bse[1],regression4.bse[1]],
          "P-value": [regression3.pvalues[1], regression4.pvalues[1]]}
    panel_B_2 = pd.DataFrame(data = params2, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_B_2.index.name = "Panel B.Baccalaureate grade: 2001-2003, Survey towns, between-school cutoffs"
    
    return [panel_B_1, panel_B_2]
    
# Panel C

def table4_panelC(data1, ik_list_table4):
    
    # Panel C. Baccalaureate grade: 2001-2003 cohorts -- between school cutoffs, IV specification
    # All towns
    
    data_for_demeaning1 = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < 1),
                                 ["bcg","agus","dga","dzag","dzag_after","sid2","uazY"]].dropna()
    dataframe1 = demean(formula = "bcg ~ agus + dzag + dzag_after + dga",
                        data = data_for_demeaning1,
                        absorb = "uazY")
    regression1 = mt.ivreg(df = dataframe1, 
                           y_name = "bcg", x_name = "agus", z_name = "dga", 
                           w_name = ["dzag", "dzag_after"])
    data_for_demeaning2 = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < ik_list_table4[4]),
                              ["bcg","agus","dga","dzag","dzag_after","sid2","uazY"]].dropna()
    dataframe2 = demean(formula = "bcg ~ agus + dzag + dzag_after + dga",
                        data = data_for_demeaning2 ,
                        absorb = "uazY")
                           
    regression2 = mt.ivreg(df = dataframe2, 
                           y_name = "bcg", x_name = "agus", z_name = "dga", 
                           w_name = ["dzag", "dzag_after"])
                           
    params1 = {"Average school transition grade": [regression1.beta[0], regression2.beta[0]],
               "Standard Error": [regression1.se[0], regression2.se[0]],
               "P-value": [regression1.pt[0], regression2.pt[0]]}
    panel_C_1 = pd.DataFrame(data = params1, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_C_1.index.name = "Panel C.Baccalaureate grade: 2001-2003, All towns, between-school cutoffs, IV specification"
    # Survey towns
    data_for_demeaning3 = data1[data1["survey"]==1].loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < 1),
                                ["bcg","agus","dga","dzag","dzag_after","sid2","uazY"]].dropna()
    dataframe3 = demean(formula = "bcg ~ agus + dzag + dzag_after + dga",
                        data = data_for_demeaning3,
                        absorb = "uazY")
    regression3 = mt.ivreg(df = dataframe3, 
                           y_name = "bcg", x_name = "agus", z_name = "dga", 
                           w_name = ["dzag", "dzag_after"])
    data_for_demeaning4 = data1[data1["survey"]==1].loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < ik_list_table4[5]),
                                ["bcg","agus","dga","dzag","dzag_after","sid2","uazY"]].dropna()
    dataframe4 = demean(formula = "bcg ~ agus + dzag + dzag_after + dga",
                        data = data_for_demeaning4,
                        absorb = "uazY")
                          
    regression4 = mt.ivreg(df = dataframe4, 
                           y_name = "bcg", x_name = "agus", z_name = "dga", 
                           w_name = ["dzag", "dzag_after"])
                          
    params2 = {"Average school transition grade": [regression3.beta[0], regression4.beta[0]],
               "Standard Error": [regression3.se[0], regression4.se[0]],
               "P-value": [regression3.pt[0], regression4.pt[0]]}
    panel_C_2 = pd.DataFrame(data = params2, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_C_2.index.name = "Panel C.Baccalaureate grade: 2001-2003, Survey towns, between-school cutoffs, IV specification"
    
    return [panel_C_1, panel_C_2]

# Panel D

def table4_panelD(data3, ik_list_table4):
     # Panel D. Baccalaureate taken dummy: 2001-2003 cohorts - between track cutoffs
    # All towns
    regression1 = areg(formula = "bct ~ dga + dzag + dzag_after", 
         data = data3.loc[(data3["dzag"] != 0) & (abs(data3["dzag"]) < 1), ["bct","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression2 = areg(formula = "bct ~ dga + dzag + dzag_after", 
         data = data3.loc[(data3["dzag"] != 0) & (abs(data3["dzag"]) < ik_list_table4[6]), 
                          ["bct","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    params1 = {"R squared": [regression1.rsquared, regression2.rsquared],
          "1{Transition grade >= cutoff}": [regression1.params[1], regression2.params[1]],
          "Standard Error": [regression1.bse[1],regression2.bse[1]],
          "P-value": [regression1.pvalues[1], regression2.pvalues[1]]}
    panel_D_1 = pd.DataFrame(data = params1, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_D_1.index.name = "Panel D.Baccalaureate taken dummy: 2001-2003, All towns, between-track cutoffs"
    # Survey towns
    regression3 = areg(formula = "bct ~ dga + dzag + dzag_after", 
         data = data3[data3["survey"] == 1].loc[(data3["dzag"] != 0) & (abs(data3["dzag"]) < 1),
                                ["bct","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression4 = areg(formula = "bct ~ dga + dzag + dzag_after", 
         data = data3[data3["survey"] == 1].loc[(data3["dzag"] != 0) & (abs(data3["dzag"]) < ik_list_table4[7]),
                                ["bct","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    params2 = {"R squared": [regression3.rsquared, regression4.rsquared],
          "1{Transition grade >= cutoff}": [regression3.params[1], regression4.params[1]],
          "Standard Error": [regression3.bse[1],regression4.bse[1]],
          "P-value": [regression3.pvalues[1], regression4.pvalues[1]]}
    panel_D_2 = pd.DataFrame(data = params2, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_D_2.index.name = "Panel D.Baccalaureate taken dummy: 2001-2003, Survey towns, between-track cutoffs"
    
    return [panel_D_1, panel_D_2]
    
# Panel E

def table4_panelE(data3, ik_list_table4):
    # Panel E. Baccalaureate grade: 2001-2003 cohorts - between track cutoffs
    # All towns
    regression1 = areg(formula = "bcg ~ dga + dzag + dzag_after", 
         data = data3.loc[(data3["dzag"] != 0) & (abs(data3["dzag"]) < 1), 
                          ["bcg","dga","dzag","dzag_after","sid2","uazY"]].dropna(),
         absorb = "uazY", cluster = "sid2")
    regression2 = areg(formula = "bcg ~ dga + dzag + dzag_after", 
         data = data3.loc[(data3["dzag"] != 0) & (abs(data3["dzag"]) < ik_list_table4[8]), 
                          ["bcg","dga","dzag","dzag_after","sid2","uazY"]].dropna(),
         absorb = "uazY", cluster = "sid2")
    params1 = {"R squared": [regression1.rsquared, regression2.rsquared],
          "1{Transition grade >= cutoff}": [regression1.params[1], regression2.params[1]],
          "Standard Error": [regression1.bse[1],regression2.bse[1]],
          "P-value": [regression1.pvalues[1], regression2.pvalues[1]]}
    panel_E_1 = pd.DataFrame(data = params1, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_E_1.index.name = "Panel E.Baccalaureate grade: 2001-2003, All towns, between-track cutoffs"
    # Survey towns
    regression3 = areg(formula = "bcg ~ dga + dzag + dzag_after", 
         data = data3[data3["survey"] == 1].loc[(data3["dzag"] != 0) & (abs(data3["dzag"]) < 1),
                                ["bcg","dga","dzag","dzag_after","sid2","uazY"]].dropna(),
         absorb = "uazY", cluster = "sid2")
    regression4 = areg(formula = "bcg ~ dga + dzag + dzag_after", 
         data = data3[data3["survey"] == 1].loc[(data3["dzag"] != 0) & (abs(data3["dzag"]) < ik_list_table4[9]),
                                ["bcg","dga","dzag","dzag_after","sid2","uazY"]].dropna(),
         absorb = "uazY", cluster = "sid2")
    params2 = {"R squared": [regression3.rsquared, regression4.rsquared],
          "1{Transition grade >= cutoff}": [regression3.params[1], regression4.params[1]],
          "Standard Error": [regression3.bse[1],regression4.bse[1]],
          "P-value": [regression3.pvalues[1], regression4.pvalues[1]]}
    panel_E_2 = pd.DataFrame(data = params2, index = ["Within 1 point_cutoff", "Within IK bound"])
    panel_E_2.index.name = "Panel E.Baccalaureate grade: 2001-2003, Survey towns, between-track cutoffs"
    
    return [panel_E_1, panel_E_2]

#############################      Table 5        #################################

def table5(data1):
    # Panel A. Full sample
    # School level average transition score
    regression1 = areg(formula = "agus ~ dga + dzag + dzag_after", 
         data = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < 1), ["agus","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression2 = areg(formula = "bct ~ dga + dzag + dzag_after", 
         data = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < 1), ["bct","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression3 = areg(formula = "bcg ~ dga + dzag + dzag_after", 
         data = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < 1), ["bcg","dga","dzag","dzag_after","sid2","uazY"]].dropna(),
         absorb = "uazY", cluster = "sid2")
    params1 = {"R squared": [regression1.rsquared, regression2.rsquared, regression3.rsquared],
          "1{Transition grade >= cutoff}": [regression1.params[1], regression2.params[1], regression3.params[1]],
          "Standard Error": [regression1.bse[1],regression2.bse[1], regression3.bse[1]],
          "P-value": [regression1.pvalues[1], regression2.pvalues[1], regression3.pvalues[1]]}
    panel_A_1 = pd.DataFrame(data = params1, index = ["School-level Average transition score", 
                                                      "Bacca. taken",
                                                      "Bacca. grade"])
    data_for_demeaning4 = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < 1),
                                 ["bcg","agus","dga","dzag","dzag_after","sid2","uazY"]].dropna()
    dataframe4 = demean(formula = "bcg ~ agus + dga + dzag + dzag_after",
                        data = data_for_demeaning4,
                        absorb = "uazY")
    
    regression4 = mt.ivreg(df = dataframe4, 
                           y_name = "bcg", x_name = "agus", z_name = "dga", 
                           w_name = ["dzag", "dzag_after"])
                       
    params2 = {"1{Transition grade >= cutoff}": [regression4.beta[0]],
               "Standard Error": [regression4.se[0]],
               "P-value": [regression4.pt[0]]}
    panel_A_2 = pd.DataFrame(data = params2, index = ["Baccalaureate grade, IV specification"])
    
    
    return [panel_A_1, panel_A_2]

def table5_panelA(data1):
    panel_A_1, panel_A_2 = table5(data1)
    panel_A_1.index.name = "Panel A. Full sample"
    panel_A_2.index.name = "Panel A. Full sample"
    return [panel_A_1, panel_A_2]

def table5_panelB(data1):
    panel_A_1, panel_A_2 = table5(data1)
    panel_A_1.index.name = "Panel B. Top tercile"
    panel_A_2.index.name = "Panel B. Top tercile"
    return [panel_A_1, panel_A_2]

def table5_panelC(data1):
    panel_A_1, panel_A_2 = table5(data1)
    panel_A_1.index.name = "Panel C. Bottom tercile"
    panel_A_2.index.name = "Panel C. Bottom tercile"
    return [panel_A_1, panel_A_2]

def table5_panelD(data1):
    panel_A_1, panel_A_2 = table5(data1)
    panel_A_1.index.name = "Panel D. Towns with 4 or more schools"
    panel_A_2.index.name = "Panel D. Towns with 4 or more schools"
    return [panel_A_1, panel_A_2]

def table5_panelE(data1):
    panel_A_1, panel_A_2 = table5(data1)
    panel_A_1.index.name = "Panel E. Towns with 3 schools"
    panel_A_2.index.name = "Panel E. Towns with 3 schools"
    return [panel_A_1, panel_A_2]

def table5_panelF(data1):
    panel_A_1, panel_A_2 = table5(data1)
    panel_A_1.index.name = "Panel F. Towns with 2 schools"
    panel_A_2.index.name = "Panel F. Towns with 2 schools"
    return [panel_A_1, panel_A_2]

    
#################################          Table 6        ################################

def table6(data7, col_name, ik_list):
    if col_name == "sc_bestintown_teacherquality11":
        regression1 = areg(formula = "{} ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < 1), [col_name,"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        regression2 = areg(formula = "{} ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < ik_list), [col_name,"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        
        params1 = {"R squared": [regression1.rsquared, regression2.rsquared],
          "1{Transition grade >= cutoff}": [regression1.params[1], regression2.params[1]],
          "Standard Error": [regression1.bse[1],regression2.bse[1]],
          "P-value": [regression1.pvalues[1], regression2.pvalues[1]]}
        panel_A_1 = pd.DataFrame(data = params1, index = ["Within 1 point of cutoff", 
                                                          "Within IK bound"])
        panel_A_1.index.name = "Principals perceive their school to be the best in teacher quality"
        return panel_A_1
    else:
        regression3 = areg(formula = "{} ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < 1), [col_name,"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        regression4 = areg(formula = "{} ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < ik_list[0]), [col_name,"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        
        regression5 = areg(formula = "{}2 ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < 1), ["{}2".format(col_name),"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        
        regression6 = areg(formula = "{}2 ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < ik_list[1]), ["{}2".format(col_name),"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        
        regression7 = areg(formula = "{}11 ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < 1), ["{}11".format(col_name),"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
         
        regression8 = areg(formula = "{}11 ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < ik_list[2]), ["{}11".format(col_name),"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        
        params2 = {"R squared": [regression3.rsquared, regression4.rsquared,
                                regression5.rsquared, regression6.rsquared,
                                regression7.rsquared, regression8.rsquared],
          "1{Transition grade >= cutoff}": [regression3.params[1], regression4.params[1],
                                           regression5.params[1], regression6.params[1],
                                           regression7.params[1], regression8.params[1]],
          "Standard Error": [regression3.bse[1],regression4.bse[1],
                            regression5.bse[1], regression6.bse[1],
                            regression7.bse[1], regression8.bse[1]],
          "P-value": [regression3.pvalues[1], regression4.pvalues[1],
                     regression5.pvalues[1], regression6.pvalues[1],
                     regression7.pvalues[1], regression8.pvalues[1]]}
        panel_B_1 = pd.DataFrame(data = params2, index = ["At student/parent level, Within 1 point of cutoff", 
                                                          "At student/parent level, Within IK bound",
                                                          "At track level, Within 1 point of cutoff",
                                                          "At track level, Within IK bound",
                                                          "At School-level, Within 1 point of cutoff",
                                                          "At School-level, Within IK bound"])
        return panel_B_1
        
        
        
#################################         Table 7       #################################

def table7(data7, col_name, ik_list):
    if col_name == "sc_bestintown_parental1":
        regression1 = areg(formula = "{} ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < 1), [col_name,"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        regression2 = areg(formula = "{} ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < ik_list), [col_name,"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        
        params1 = {"R squared": [regression1.rsquared, regression2.rsquared],
          "1{Transition grade >= cutoff}": [regression1.params[1], regression2.params[1]],
          "Standard Error": [regression1.bse[1],regression2.bse[1]],
          "P-value": [regression1.pvalues[1], regression2.pvalues[1]]}
        panel_A_1 = pd.DataFrame(data = params1, index = ["Within 1 point of cutoff", 
                                                          "Within IK bound"])
        panel_A_1.index.name = "Principals perceive their school to be the best in parental participation"
        return panel_A_1
    else:
        regression3 = areg(formula = "{}1 ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < 1), ["{}1".format(col_name),"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        regression4 = areg(formula = "{}1 ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < ik_list[0]), ["{}1".format(col_name),"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        
        regression5 = areg(formula = "{}2 ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < 1), ["{}2".format(col_name),"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        
        regression6 = areg(formula = "{}2 ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < ik_list[1]), ["{}2".format(col_name),"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        
        regression7 = areg(formula = "{} ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < 1), [col_name,"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
         
        regression8 = areg(formula = "{} ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < ik_list[2]), [col_name,"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        
        params2 = {"R squared": [regression3.rsquared, regression4.rsquared,
                                regression5.rsquared, regression6.rsquared,
                                regression7.rsquared, regression8.rsquared],
          "1{Transition grade >= cutoff}": [regression3.params[1], regression4.params[1],
                                           regression5.params[1], regression6.params[1],
                                           regression7.params[1], regression8.params[1]],
          "Standard Error": [regression3.bse[1],regression4.bse[1],
                            regression5.bse[1], regression6.bse[1],
                            regression7.bse[1], regression8.bse[1]],
          "P-value": [regression3.pvalues[1], regression4.pvalues[1],
                     regression5.pvalues[1], regression6.pvalues[1],
                     regression7.pvalues[1], regression8.pvalues[1]]}
        panel_B_1 = pd.DataFrame(data = params2, index = ["At School level, Within 1 point of cutoff", 
                                                          "At School-level, Within IK bound",
                                                          "At track level, Within 1 point of cutoff",
                                                          "At track level, Within IK bound",
                                                          "At student/parent level, Within 1 point of cutoff",
                                                          "At student/parent level, Within IK bound"])
        return panel_B_1
        

################################          Table 8        ####################################

def table8(data7, col_name, ik_list):
    if col_name == "sc_bestintown_studentquality1":
        regression1 = areg(formula = "{} ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < 1), [col_name,"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        regression2 = areg(formula = "{} ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < ik_list), [col_name,"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        
        params1 = {"R squared": [regression1.rsquared, regression2.rsquared],
          "1{Transition grade >= cutoff}": [regression1.params[1], regression2.params[1]],
          "Standard Error": [regression1.bse[1],regression2.bse[1]],
          "P-value": [regression1.pvalues[1], regression2.pvalues[1]]}
        panel_A_1 = pd.DataFrame(data = params1, index = ["Within 1 point of cutoff", 
                                                          "Within IK bound"])
        panel_A_1.index.name = "Principals perceive their school to be the best in parental participation"
        return panel_A_1
    else:
        regression3 = areg(formula = "{}1 ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < 1), ["{}1".format(col_name),"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        regression4 = areg(formula = "{}1 ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < ik_list[0]), ["{}1".format(col_name),"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        
        regression5 = areg(formula = "{}2 ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < 1), ["{}2".format(col_name),"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        
        regression6 = areg(formula = "{}2 ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < ik_list[1]), ["{}2".format(col_name),"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        
        regression7 = areg(formula = "{} ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < 1), [col_name,"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
         
        regression8 = areg(formula = "{} ~ dga + dzag + dzag_after".format(col_name), 
         data = data7.loc[(data7["dzag"] != 0) & (abs(data7["dzag"]) < ik_list[2]), [col_name,"dga",
                                                                            "dzag","dzag_after","usY","uazY"]].dropna(),
         absorb = "uazY", cluster = "usY")
        
        params2 = {"R squared": [regression3.rsquared, regression4.rsquared,
                                regression5.rsquared, regression6.rsquared,
                                regression7.rsquared, regression8.rsquared],
          "1{Transition grade >= cutoff}": [regression3.params[1], regression4.params[1],
                                           regression5.params[1], regression6.params[1],
                                           regression7.params[1], regression8.params[1]],
          "Standard Error": [regression3.bse[1],regression4.bse[1],
                            regression5.bse[1], regression6.bse[1],
                            regression7.bse[1], regression8.bse[1]],
          "P-value": [regression3.pvalues[1], regression4.pvalues[1],
                     regression5.pvalues[1], regression6.pvalues[1],
                     regression7.pvalues[1], regression8.pvalues[1]]}
        panel_B_1 = pd.DataFrame(data = params2, index = ["At School level, Within 1 point of cutoff", 
                                                          "At School-level, Within IK bound",
                                                          "At track level, Within 1 point of cutoff",
                                                          "At track level, Within IK bound",
                                                          "At student/parent level, Within 1 point of cutoff",
                                                          "At student/parent level, Within IK bound"])
        return panel_B_1
        
        
        
######################################  Extension Part Tables  #####################################
# Table 5 with IK bounds
def extension_table5(data1, ik_list):
    # Panel A. Full sample
    # School level average transition score
    regression1 = areg(formula = "agus ~ dga + dzag + dzag_after", 
         data = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < ik_list[0]), ["agus","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression2 = areg(formula = "bct ~ dga + dzag + dzag_after", 
         data = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < ik_list[1]), ["bct","dga","dzag","dzag_after","sid2","uazY"]],
         absorb = "uazY", cluster = "sid2")
    regression3 = areg(formula = "bcg ~ dga + dzag + dzag_after", 
         data = data1.loc[(data1["dzag"] != 0) & (abs(data1["dzag"]) < ik_list[2]),
                          ["bcg","dga","dzag","dzag_after","sid2","uazY"]].dropna(),
         absorb = "uazY", cluster = "sid2")
    params1 = {"R squared": [regression1.rsquared, regression2.rsquared, regression3.rsquared],
          "1{Transition grade >= cutoff}": [regression1.params[1], regression2.params[1], regression3.params[1]],
          "Standard Error": [regression1.bse[1],regression2.bse[1], regression3.bse[1]],
          "P-value": [regression1.pvalues[1], regression2.pvalues[1], regression3.pvalues[1]]}
    panel_A_1 = pd.DataFrame(data = params1, index = ["School-level Average transition score", 
                                                      "Bacca. taken",
                                                      "Bacca. grade"])
 
    
    
    return panel_A_1

def ext_table5_panelA(data1, ik_list):
    panel_A_1 = extension_table5(data1, ik_list)
    panel_A_1.index.name = "Panel A. Full sample, with IK bounds"
    return panel_A_1

def ext_table5_panelB(data1, ik_list):
    panel_A_1 = extension_table5(data1, ik_list)
    panel_A_1.index.name = "Panel B. Top tercile, with IK bounds"
    
    return panel_A_1

def ext_table5_panelC(data1, ik_list):
    panel_A_1 = extension_table5(data1, ik_list)
    panel_A_1.index.name = "Panel C. Bottom tercile, with IK bounds"
 
    return panel_A_1

def ext_table5_panelD(data1, ik_list):
    panel_A_1 = extension_table5(data1, ik_list)
    panel_A_1.index.name = "Panel D. Towns with 4 or more schools, with IK bounds"
    
    return panel_A_1

def ext_table5_panelE(data1, ik_list):
    panel_A_1 = extension_table5(data1, ik_list)
    panel_A_1.index.name = "Panel E. Towns with 3 schools, with IK bounds"
    
    return panel_A_1

def ext_table5_panelF(data1, ik_list):
  
    panel_A_1 = extension_table5(data1, ik_list)
    panel_A_1.index.name = "Panel F. Towns with 2 schools, with IK bounds"
   
    return panel_A_1


        