from flask import Flask, render_template, request
from flask import jsonify
import json

import h2o

h2o.init(min_mem_size=8)
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.model.metrics_base import H2OAutoEncoderModelMetrics
from h2o.model.metrics_base import MetricsBase
from h2o.model.metrics_base import H2OBinomialModelMetrics
from h2o.model.metrics_base import H2OClusteringModelMetrics
from h2o.model.metrics_base import H2ODimReductionModelMetrics
from h2o.model.metrics_base import H2OMultinomialModelMetrics
import pandas as pd
import h2o
from h2o.job import H2OJob
from h2o.frame import H2OFrame
from h2o.exceptions import H2OValueError
from h2o.estimators.estimator_base import H2OEstimator
from h2o.two_dim_table import H2OTwoDimTable
from h2o.display import H2ODisplay
from h2o.grid.metrics import *  # NOQA
from h2o.utils.backward_compatibility import backwards_compatible
from h2o.utils.shared_utils import deprecated, quoted
from h2o.utils.compatibility import *  # NOQA
from h2o.utils.typechecks import assert_is_type, is_type

h2o.no_progress()

app = Flask(__name__)


# Here the method is not mensioned.That means we r not using any inputs given by user here.
# @app.route('/')
# def student():
#   return render_template('student.html')
#////////////////////
def case1(I, loan, r):
    n = 50 / 4
    # installment is A
    payment = (loan * (r / (12 * 100)) * (((r / (12 * 100)) + 1) ** n)) / ((((r / (12 * 100)) + 1) ** n) - 1)
    disposible = I * 0.9
    if disposible >= payment:
        return payment
    elif disposible < payment:
        return disposible * 0.9

    # district ,lbr , age


def case2(monthlyPayable, age):
    n = 50 / 4
    if age <= 50.0:
        return monthlyPayable * n
    elif age < 60.0:
        return monthlyPayable * n * 0.9
    elif age >= 60.0:
        return monthlyPayable * n * 0.8

    #  in (3, 6, 34, 47, 54, 55, 62, 67)


def case3(approved_ln_amt, lbr):
    if lbr in (3, 6, 34, 47, 54, 55, 62, 67):
        return approved_ln_amt * 0.9
    else:
        return approved_ln_amt


#/////////////////////


##Here the method is mensioned.That means we r using any inputs given by user here.
@app.route('/result', methods=['POST'])
def JsonHandler():
    # get the built model from training set
    saved_model = h2o.load_model("Grid_XGBoost_py_4_sid_b12f_model_python_1547186079651_1_model_1")

    #////////////////////////////////////////////////

    #/////////////////////////////////////////////////
    # Get Json object
    content = [request.get_json()]

    # convert json object to  pandas dataframe
    df = pd.DataFrame.from_dict(content, orient='columns')

    # convert pandas data frame to h2o input
    df_test = h2o.H2OFrame(df)
    # print(df_test)
    # ///////////////////////////////////
    preds = saved_model.predict(df_test)
    dff = preds.as_data_frame();
    dff
    # d= dff.to_json()
    # print(d)

    # =====================================================new part

    p1 = df['IncomeCustDisposable'].values.astype('float64')
    p2 = df['AppldLoanAmt'].values.astype('float64')
    p3 = df['IntRate'].values.astype('float64')

    df['monthlyPayable'] = case1(p1, p2, p3)
    print(df['monthlyPayable'])
    p4 = df['monthlyPayable'].values.astype('float64')
    p5 = df['Age'].values.astype('float64')
    p6 = df['LBrCode'].values

    df['approved_ln_amt'] = case2(p4, p5)


    p7 = df['approved_ln_amt'].values.astype('float64')

    df['approved_ln_amt1'] = case3(p7, p6)
    print(df['approved_ln_amt1'])
# #

    p8 = df['approved_ln_amt1'].values.astype('float64')



    df['loan_amount'] = case4(df['approved_ln_amt1'],dff)
    p10 = df['loan_amount'].values.astype('float64')
    # df['Customer_Group']= cust_grp(p10, p2)
    # print(df['Customer_Group'])
    df_result = pd.DataFrame.from_dict(cust_grp(p10, p2))
    # , 'credit_score', 'amount'] = cust_grp(p10, p2)
    # print(df['Customer_Group', 'credit_score', 'amount'])
    result = df_result.to_json()
    print(result)
# /////////////////////////////////////
    return jsonify(result)

#///////////////////////
def case4(approved_ln_amt,dff):
    print("case444444")
    p9 = dff['predict'].values
    print(dff['predict'])
    if p9 == 'A':
        return approved_ln_amt
    elif p9 == 'B':
        return approved_ln_amt * 0.9
    elif p9 == 'C':
        print("im hereee")
        return approved_ln_amt * 0.85
    elif p9 == 'D':
        return approved_ln_amt * 0.8


def cust_grp(approved, applied):
    credit_score = approved / applied
    amount = approved
    if credit_score >= 0.99:
        grp = "A"
        return {'Customer_Group': grp, 'Credit Score': credit_score, 'Loan Amount': amount}
    elif credit_score >= 0.7:
        grp = "B"
        return {'Customer_Group': grp, 'Credit Score': credit_score, 'Loan Amount': amount}

    elif credit_score >= 0.4:
        grp = "C"
        return {'Customer_Group': grp, 'Credit Score': credit_score, 'Loan Amount': amount}

    elif credit_score < 0.4:
        grp = "D"
        return {'Customer_Group': grp, 'Credit Score': credit_score, 'Loan Amount': amount}
#///////////////////

if __name__ == '__main__':
    app.run(debug=True)
