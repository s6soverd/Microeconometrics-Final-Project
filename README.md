
# Replication of Pop-Eleches, Cristian, and Miguel Urquiola (2013)
---
<a href="https://nbviewer.jupyter.org/github/HumanCapitalAnalysis/microeconometrics-course-project-s6soverd/blob/master/replication.ipynb"
   target="_parent">
   <img align="center"
  src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png"
      width="109" height="20">
</a>
<a href="https://mybinder.org/v2/gh/HumanCapitalAnalysis/microeconometrics-course-project-s6soverd/master?filepath=replication.ipynb"
    target="_parent">
    <img align="center"
       src="https://mybinder.org/badge_logo.svg"
       width="109" height="20">
</a>
---

This repository contains my replication of results from Pop-Eleches, Cristian, and Miguel Urquiola (2013). "Going to a Better School: Effects and Behavioral Responses." 
*American Economic Review*, 103 (4): 1289-1324. The original paper, including the data sets, and the codes by the authors can be accessed [here](https://www.aeaweb.org/articles?id=10.1257/aer.103.4.1289).


### Brief Description of the paper:


Pop-Eleches et al. (2013) examines the effect of going to a better school on student outcomes and on behavioral responses that amplify or reduce the quality of educational quality. They apply **regression discontinuity design** to the Romanian secondary school system, producing two findings. First, students who gain access to higher achievement schools perform better in a graduation test. Secondly, the opportunity to attend quality-wise better schools result in significant behavioral responses, particularly: *(i)* Teachers sort in a manner consistent with a preference for higher achieving students, *(ii)* Children who get into better schools realize they are relatively weaker and feel marginalized; *(iii)* Parents reduce their effort when their children attend a better school. \
In this project, I replicate the results from Pop-Eleches et al. (2013) and look into the causal relationship between attending a higher-ranked school and student outcomes, and 
behavioral responses it triggers. 


### On a Side Note

The replication is conducted using Python, I have stored my functions that generate tables and figures in seperate files, which you can access
[here](https://github.com/HumanCapitalAnalysis/microeconometrics-course-project-s6soverd/tree/master/auxiliary/project_auxiliary_tables.py),and
[here](https://github.com/HumanCapitalAnalysis/microeconometrics-course-project-s6soverd/tree/master/auxiliary/project_auxiliary_plots.py), 


### References:

* Pop-Eleches, Cristian, and Miguel Urquiola. 2013. "Going to a Better School: Effects and Behavioral Responses." American Economic Review, 103 (4): 1289-1324.


## Reproducibility

To ensure full reproducibility of your project, please try to set up a [Travis CI](https://travis-ci.org) as your continuous integration service. An introductory tutorial for [conda](https://conda.io) and [Travis CI](https://docs.travis-ci.com/) is provided [here](https://github.com/HumanCapitalAnalysis/template-course-project/blob/master/tutorial_conda_travis.ipynb). While not at all mandatory, setting up a proper continuous integration workflow is an extra credit that can improve the final grade.

[![Build Status](https://travis-ci.org/HumanCapitalAnalysis/template-course-project.svg?branch=master)](https://travis-ci.org/HumanCapitalAnalysis/template-course-project)

In some cases you might not be able to run parts of your code on  [Travis CI](https://travis-ci.org) as, for example, the computation of results takes multiple hours. In those cases you can add the result in a file to your repository and load it in the notebook. See below for an example code.

```python
# If we are running on TRAVIS-CI we will simply load a file with existing results.
if os.environ['TRAVIS']:
  rslt = pkl.load(open('stored_results.pkl', 'br'))
else:
  rslt = compute_results()

# Now we are ready for further processing.
...
```




[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/HumanCapitalAnalysis/template-course-project/blob/master/LICENSE)
[![Continuous Integration](https://github.com/HumanCapitalAnalysis/template-course-project/workflows/Continuous%20Integration/badge.svg)](https://github.com/HumanCapitalAnalysis/template-course-project/actions)
