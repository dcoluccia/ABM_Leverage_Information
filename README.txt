Coluccia, Davide M.	Hill, J.	Nahai-Williamson, P. 
"Back to Basics: Rationality, heterogeneity and leverage in financial markets". 
(2018) Working Paper

Note: code written in Python 3. Additional modules: SciPy, NumPy, Matplotlib, Pandas, Mpl_toolkits.

REPLICATION CODE: Instructions
The main.py file is executed by the call_.py files. The output of each call_.py file is a set of figures and time series which can be easily stored and saved.
Simply run a call_.py file by clicking on it. To modify the baseline calibration, open the call_.py file and customize it directly from the preamble.
To run the perfect rationality simulation for each call_.py, simply set gamma=1 in the calibration.
To run the permanent shock simulation un-comment the permanent shock method from the main.py file and comment the temporary shock one.
-> call_1corporateshock: figures 10,11,12 (execution time: 40'')
-> call_cetpar: figures 8,9 (execution time: 3')
-> call_cetpar_expectations: figure 6 (execution time: 25'')
-> call_cetpar_leverage: figure 7 (execution time: 25'')
-> call_generaleconomy: figures 1,2,3,4,5,13,14 (execution time: 30'')
-> call_mc_cetpar: figure 16 (execution time: 2h)
-> call_mc_generaleconomy: figure 15 (execution time: 5'')


