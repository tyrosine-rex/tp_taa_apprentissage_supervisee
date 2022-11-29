##WORKFLOW STUFF
from hashlib import sha1
from pickle import dump, load 
from sys import stderr
from os.path import exists
from os import makedirs

def cache_pickle(func):
    def wrapper(*args, **kwargs):
        cache_dir = f"./.cache_pickle/{func.__name__}"
        
        if not exists(cache_dir):
            makedirs(cache_dir)
            
        tohash = f"{args}{kwargs}"
        basename = sha1(tohash.encode('utf-8')).hexdigest()
        pkl_path = f"{cache_dir}/{basename}.pkl"

        if exists(pkl_path): 
            res = load(open(pkl_path, "rb"))
            print(f"Loaded from {pkl_path}", file=stderr)
        else:
            res = func(*args, **kwargs)
            dump(res, open(pkl_path, "wb"))
            print(f"Saved to {pkl_path}", file=stderr)            
        return res
    return wrapper


## SK-LEARN STUFF
import pandas as pd
import numpy as np
from IPython.display import Markdown, display

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_validate, KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


CLASSIFIERS = { 
    'RF_10': RandomForestClassifier(n_estimators=10, random_state=1),
    'RF_20': RandomForestClassifier(n_estimators=20, random_state=1),
    'RF_40': RandomForestClassifier(n_estimators=40, random_state=1),
    'RF_80': RandomForestClassifier(n_estimators=80, random_state=1),
    'RF_120': RandomForestClassifier(n_estimators=120, random_state=1),
    'RF_160': RandomForestClassifier(n_estimators=160, random_state=1),

    "NB": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(),

    "MLP_20_10" : MLPClassifier(hidden_layer_sizes=(20,10), random_state=1),
    "MLP_40_20" : MLPClassifier(hidden_layer_sizes=(40,20), random_state=1),
    "MLP_60_30" : MLPClassifier(hidden_layer_sizes=(60,30), random_state=1),
    "MLP_100_50" : MLPClassifier(hidden_layer_sizes=(100,50), random_state=1),

    "CART_gini_5" : DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=1),
    "CART_gini_10" : DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=1),
    "CART_gini_15" : DecisionTreeClassifier(criterion="gini", max_depth=15, random_state=1),

    "CART_entropy_5" : DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=1),
    "CART_entropy_10" : DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=1),
    "CART_entropy_15" : DecisionTreeClassifier(criterion="entropy", max_depth=15, random_state=1),

    "KNN_10" : KNeighborsClassifier(n_neighbors=10),
    "KNN_20" : KNeighborsClassifier(n_neighbors=20),
    "KNN_40" : KNeighborsClassifier(n_neighbors=40),
    "KNN_80" :  KNeighborsClassifier(n_neighbors=80),
    "KNN_120" :  KNeighborsClassifier(n_neighbors=120),
    "KNN_160" :  KNeighborsClassifier(n_neighbors=160),

    "Bag_10" : BaggingClassifier(n_estimators=10, random_state=1),
    "Bag_20" : BaggingClassifier(n_estimators=20, random_state=1),
    "Bag_40" : BaggingClassifier(n_estimators=40, random_state=1),
    "Bag_80" : BaggingClassifier(n_estimators=80, random_state=1),
    "Bag_120" : BaggingClassifier(n_estimators=120, random_state=1),
    "Bag_160" : BaggingClassifier(n_estimators=160, random_state=1),

    "AdaB_20" : AdaBoostClassifier(n_estimators=10, random_state=1),
    "AdaB_20" : AdaBoostClassifier(n_estimators=20, random_state=1),
    "AdaB_40" : AdaBoostClassifier(n_estimators=40, random_state=1),
    "AdaB_80" : AdaBoostClassifier(n_estimators=80, random_state=1),
    "AdaB_120" : AdaBoostClassifier(n_estimators=120, random_state=1),
    "AdaB_160" : AdaBoostClassifier(n_estimators=160, random_state=1)
}

CLASSIFIERS_LIGHT = { 
    'RF_10': RandomForestClassifier(n_estimators=10, random_state=1),
    'RF_20': RandomForestClassifier(n_estimators=20, random_state=1),
    'RF_40': RandomForestClassifier(n_estimators=40, random_state=1),

    "NB": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(),

    "MLP_20_10" : MLPClassifier(hidden_layer_sizes=(20,10), random_state=1),
    "MLP_40_20" : MLPClassifier(hidden_layer_sizes=(40,20), random_state=1),

    "CART_gini_5" : DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=1),
    "CART_gini_10" : DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=1),
    "CART_gini_15" : DecisionTreeClassifier(criterion="gini", max_depth=15, random_state=1),

    "CART_entropy_5" : DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=1),
    "CART_entropy_10" : DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=1),
    "CART_entropy_15" : DecisionTreeClassifier(criterion="entropy", max_depth=15, random_state=1),

    "KNN_10" : KNeighborsClassifier(n_neighbors=10),
    "KNN_20" : KNeighborsClassifier(n_neighbors=20),
    "KNN_40" : KNeighborsClassifier(n_neighbors=40),
    "KNN_80" :  KNeighborsClassifier(n_neighbors=80),

    "Bag_10" : BaggingClassifier(n_estimators=10, random_state=1),
    "Bag_20" : BaggingClassifier(n_estimators=20, random_state=1),
    "Bag_40" : BaggingClassifier(n_estimators=40, random_state=1),

    "AdaB_20" : AdaBoostClassifier(n_estimators=10, random_state=1),
    "AdaB_20" : AdaBoostClassifier(n_estimators=20, random_state=1),
    "AdaB_40" : AdaBoostClassifier(n_estimators=40, random_state=1)
}


def test_model(X_train, Y_train, X_test, Y_test, sk_fun, comment="", **kwargs):

    COLS_EVAL = ["method", "comment", "precision", "accuracy", "recall", "VP", "VN", "FP", "FN", "args"]
    
    model = sk_fun(**kwargs)            # call a model
    model.fit(X_train, Y_train)         # training
    Y_pred = model.predict(X_test)      # testing
    
    # metrics
    pr, acc, rec = precision_score(Y_test, Y_pred), accuracy_score(Y_test, Y_pred), recall_score(Y_test, Y_pred)
    Conf = confusion_matrix(Y_test, Y_pred)

    # maj df_eval
    line = [sk_fun.__name__, comment, pr, acc, rec, Conf[1, 1], Conf[0, 0], Conf[0, 1], Conf[1, 0], [str(kwargs)]]
    line_df = pd.DataFrame({k:v for k, v in zip(COLS_EVAL, line)}, columns=COLS_EVAL)

    # display confusion matrix
    display(Markdown(f"Confusion matrix with {sk_fun.__name__} {comment} {str(kwargs)}"))
    display(pd.DataFrame(Conf, index=["is 0", "is 1"], columns=["predicted 0", "predicted 1"]))
    
    return line_df


@cache_pickle
def find_best_param(sk_function, params, X_train, Y_train):
    gs = GridSearchCV(sk_function(), params, scoring="precision") #we target the best precision score)
    gs.fit(X_train, Y_train)
    display(Markdown(f"Best params for '{sk_function.__name__}' is :\n{gs.best_params_}\nscore (precision): {gs.best_score_}\n"))
    return gs


@cache_pickle
def make_pipeline(X_train, Y_train, sk_function, TRIM_COLS, params):
    # describe the pipeline
    trim = ColumnTransformer([("trim", "passthrough", TRIM_COLS)]) # trim data with 7 best variables
    # StdScaling data
    trim_ss = Pipeline([
        ("trim", trim), 
        ('stdScale', StandardScaler())
    ])
    # StdScaling data then get 3 first PCA axis 
    trim_ss_pca3 = Pipeline([
        ("trim", trim), 
        ('stdScale', StandardScaler()), 
        ('pca3comp', PCA(n_components=3))
    ])
    # bind 3 frist PCA axis to the scaled data
    prep_data = FeatureUnion([
        ('trim + stdScale', trim_ss),
        ('trim + stdScale + pca3comp', trim_ss_pca3)
    ]) 
    # then add estimator
    pipeline = Pipeline([
        ('preprocess', prep_data), 
        ('model', sk_function(**params))
    ]) 
    # train the pipeline
    pipeline.fit(X_train, Y_train)

    return pipeline


def test_pipeline(pipelines, X_test, Y_test):
    COLS_EVAL_2 = ["method", "accuracy", "precision", "recall", "params"]
    df = pd.DataFrame(columns=COLS_EVAL_2)
    for pipe in pipelines:
        Y_pred = pipe.predict(X_test)
        dict_line = {
            "method" : pipe.named_steps['model'].__class__.__name__,
            "accuracy" : pipe.score(X_test, Y_test),
            "precision" : precision_score(Y_test, Y_pred),
            "recall" : recall_score(Y_test, Y_pred),
            "params" : [pipe.named_steps['model']]
        }
        df_line = pd.DataFrame(dict_line, columns=COLS_EVAL_2)
        df = pd.concat([df, df_line], ignore_index=True)
    return df


@cache_pickle
def run_classifiers(sk_func_dict, X, Y, scores=['accuracy', 'precision', 'roc_auc'], nb_split=10):
    COLS_EVAL_3 = [
        'method', 'test_precision_mean', 'test_precision_sd', 
        'test_accuracy_mean', 'test_accuracy_sd','test_roc_auc_mean', 'test_roc_auc_sd',
        'fit_time_mean', 'fit_time_sd', 'score_time_mean', 'score_time_sd',
    ]
    df = pd.DataFrame(columns=COLS_EVAL_3)
    kf = KFold(n_splits=nb_split, shuffle=True, random_state=0)
    size_clfs = len(sk_func_dict)
    for idx, method in enumerate(sk_func_dict):
        print(f"({idx+1}/{size_clfs}): {method} is running!", end="", file=stderr)
        
        cv_metrics = cross_validate(sk_func_dict[method], X, Y, cv=kf, scoring=scores)
        
        dict_metrics = {i:{'mean':np.mean(v), 'sd':np.std(v)} for i, v in cv_metrics.items()}
        dict_metrics["method"] = method
        df_metrics = pd.json_normalize(dict_metrics, sep="_")
        df = pd.concat([df, df_metrics], ignore_index=True)
        print("...   DONE", file=stderr)
    return df


@cache_pickle
def test_trimming(X_train, Y_train, X_test, Y_test, sk_fun_dict, index_by_importance):
    n = X_train.shape[1]+1
    scores = {k:np.zeros(n) for k in sk_fun_dict}
    for k, model in sk_fun_dict.items():
        for f in range(n):
            Sub_train = X_train[:, index_by_importance[:f+1]]
            Sub_test = X_test[:, index_by_importance[:f+1]]
            model.fit(Sub_train, Y_train)
            Y_pred = model.predict(Sub_test)
            scores[k][f] = np.round(accuracy_score(Y_test,Y_pred), 3)
    return scores


## PLOTS STUFFS
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D
from matplotlib import colormaps
from matplotlib.patches import Patch
import matplotlib.pyplot as plt


def __column_typer(col) :
    modals = col.value_counts().index
    is_int_cls = lambda x: True if isinstance(x, (np.integer, int)) else False
    is_float_cls = lambda x: True if isinstance(x, (np.floating, float)) else False
    is_str_cls = lambda x: True if isinstance(x, str) else False

    if len(modals) == 2:
        return "binary"
    elif any(list(map(is_float_cls, modals))):
        if all(list(map(float.is_integer, modals))):
            return "integer"
        else:
            return "float"
    elif all(list(map(is_int_cls, modals))):
        return "integer"
    elif all(list(map(is_str_cls, modals))):
        return "string"
    else:
        return "other"


def explore_data(df, name, DPI=226):
    COLORS = {
        "float"     : colormaps['tab20'].colors[:2],
        "integer"   : colormaps['tab20'].colors[2:4],
        "string"    : colormaps['tab20'].colors[4:6],
        "binary"    : colormaps['tab20'].colors[8:10],
        "NA"        : colormaps['tab20'].colors[6]
    }

    # legend elements
    red_txt = Line2D([], [], marker=MarkerStyle("$123$"),
                color=COLORS["NA"],linestyle='None', markersize=14)
    black_txt = Line2D([], [], marker=MarkerStyle("$123$"), 
                color="black", linestyle='None', markersize=14)
    f1, f2 = Patch(facecolor=COLORS["float"][0]), Patch(facecolor=COLORS["float"][1])
    i1, i2 = Patch(facecolor=COLORS["integer"][0]), Patch(facecolor=COLORS["integer"][1])
    s1, s2 = Patch(facecolor=COLORS["string"][0]), Patch(facecolor=COLORS["string"][1])
    b1, b2 = Patch(facecolor=COLORS["binary"][0]), Patch(facecolor=COLORS["binary"][1])
    n1 = Patch(facecolor=COLORS["NA"])
    null = Patch(alpha=0)

    # ax.legend arguments
    lgd_args = {
        "loc" :                 "right",
        "ncol" :                2,
        "bbox_to_anchor" :     (1.32, 0.5),
        "columnspacing" :        -0.5,
        "handlelength" :        1.0,
        "handletextpad" :       0.5,
        "framealpha" : 1, 
        "edgecolor" : "black"
    }

    # ax.text arguments
    txt_args = {
        "horizontalalignment" : "center",
        "fontweight" : "bold",
        "fontsize" : "small"
    }

    #new plot
    fig, ax = plt.subplots(dpi=DPI)
    xmax = df.shape[0]
        
    #populate the bar plot
    for key, serie in df.items():
        
        unique = serie.value_counts(dropna=False).sort_values(ascending=False)
        cumul = 0
        type = __column_typer(serie)
 
        for idx, (modalite, count) in enumerate(unique.items()): 
           
            if pd.isna(modalite):
                c = COLORS["NA"]
            elif type in COLORS.keys():
                c = COLORS[type][0] if idx%2 == 0 else COLORS[type][1]
            else:
                print(f"err with col:{key} => {idx}, ({modalite}, {count}, {type(modalite)}")
        
            ax.barh(y=str(key), width=count, left=cumul, color=c)
            cumul += count

        ax.text(xmax*1.05, key, unique.index.notna().sum(), **txt_args)
        ax.text(xmax*1.15, key, serie.isna().sum(), color=COLORS["NA"], **txt_args)
    
    # adjust the plot
    ax.set_xlim((0, xmax*1.2))
    ax.set_title(f"Diversity of value per column in '{name}'")
    ax.set_xlabel("Rows")
    ax.set_ylabel("Columns/variables")

    # set legend
    ax.legend(  
        handles=[f1, i1, s1, b1, n1, null, null,
                 f2, i2, s2, b2,n1, black_txt, red_txt],
        labels=['', '', '', '', '', '', '',
                'continuous', 'discrete', 'categorical', 'binary','NA', 'unique values', 'NA values'], 
        **lgd_args)


def comparative_preprocessing(df, scores=["precision", "accuracy", "recall"], DPI=226):
    PREPRO = {
        "" : 0, 
        "MinMaxScale" : 1,
        "StdScale" : 2, 
        "StdScale + PCA" : 3,
        "Trim + StdScale": 4
    }
    fig, ax = plt.subplots(ncols=len(scores), figsize=(12, 4), dpi=DPI)
    for i, sc in enumerate(scores):
        
        ax[i].set_xticks(
            list(range(5)),
            labels=["None", "MinMax", "Std", "Std + PCA", "Trim + Std"],
            rotation=45
        )
        ax[i].set_title(f"'{sc}' by different preprocessing", size=12)

        for met in set(df["method"]):
            dat = df[df["method"] == met]
            dat["x"] = list(map(lambda x:PREPRO[x], dat["comment"]))
            dat = dat.sort_values(by="x")
            ax[i].plot(dat["x"], dat[sc], label=met)

    hd, lb = ax[0].get_legend_handles_labels()    
    fig.legend(
        hd, lb, 
        loc='right',
        bbox_to_anchor=(1.08, 0.5), 
        framealpha = 1, 
        edgecolor = "black"
    )


def results_run_clfs(
    run_df, run_name, top=10, height=5, DPI=226, 
    scores=["test_accuracy", "test_roc_auc", "test_precision"]
):

    COLOR_CLFS = {
        "RF"    : colormaps["tab10"].colors[0],
        "Bag"   : colormaps["tab10"].colors[1], 
        "AdaB"  : colormaps["tab10"].colors[2],
        "NB"    : colormaps["tab10"].colors[3],
        "MLP"   : colormaps["tab10"].colors[4],
        "CART"  : colormaps["tab10"].colors[5],
        "KNN"   : colormaps["tab10"].colors[6],
        "QDA"   : colormaps["tab10"].colors[7]
    }
    fig, ax = plt.subplots(ncols=len(scores), figsize=(10, height), dpi=DPI)

    for i, sc in enumerate(scores): # len(scores) loop
        mean, sd = f"{sc}_mean", f"{sc}_sd"

        idx = run_df.sort_values(by=mean, ascending=False).index[:top]
        xmin, xmax = run_df[mean][idx].tolist()[-1], run_df[mean][idx].tolist()[0]
        xmin, xmax = xmin*0.9, xmax*1.1
        vmax, vlast, vmin = run_df[mean].max(), run_df[mean][idx].min(), run_df[mean].min()
        
        ax[i].set_xlim([xmin, xmax])
        ax[i].tick_params(axis='both', labelsize='xx-small')
        ax[i].set_title(f"'{sc}' max:{vmax:.3f} last:{vlast:.3f} min:{vmin:.3f}", fontsize='x-small')
      
        for r in idx[::-1]: # n methods loop
            serie = run_df.iloc[r]
            c = COLOR_CLFS[serie["method"].split("_")[0]]
            ax[i].barh(serie["method"], serie[mean], xerr=serie[sd], color=c)

    fig.suptitle(f"Top {top} methods for {run_name}", fontsize=16)
    fig.tight_layout()
    fig.legend(
        [Patch(facecolor=COLOR_CLFS[key]) for key in COLOR_CLFS],[key for key in COLOR_CLFS],
        fontsize='small', 
        frameon=False, 
        bbox_to_anchor=(1.1, 0.75)
    )


def results_test_trimming(res):
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    for i, k in enumerate(res):
        ax[i].plot(res[k])
        ax[i].axvline(np.argmax(res[k]), color="red")
        ax[i].set_title(f"'{k}'")
        ax[i].set_ylabel("Accuracy")
        ax[i].set_xlabel("Nombre de variables incluses")
        ax[i].set_xticks(
                list(range(len(res[k]))),
                labels=[str(i) for i in range(len(res[k]))]
            )
    fig.suptitle("Evolution de 'accuracy' en fonction des variables")
    fig.tight_layout()


    