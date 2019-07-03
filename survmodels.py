
import rpy2
import pandas as pd
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
import matplotlib.pyplot as plt
import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sksurv.metrics import concordance_index_censored
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
R = ro.r
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import globalenv
glmnet = importr('glmnet')
survival = importr('survival')
ranger=importr('ranger')
CoxBoost=importr('CoxBoost')
Hmisc=importr('Hmisc')
from rpy2.robjects.packages import STAP

class models:
   
    @staticmethod 
    def fitcoxph(x,y):
        """
        Fit a cox proportional hazards model
        Input : data_x : Data frame in pandas with covariates 
                  y: Survival time array with time and censored event 
        Output : Coefficients with p-value dataframe
                 Concordance and standard error 
        """
        def r_matrix_to_data_frame(r_matrix):
            """
            Convert an R matrix into a Pandas DataFrame
            """
            from rpy2.robjects import pandas2ri
            array = pandas2ri.ri2py(r_matrix)
            return pd.DataFrame(array,
                        index=r_matrix.names[0],
                        columns=r_matrix.names[1])
        pandas2ri.activate()
        x_raw = pandas2ri.py2ri(x)
        y_raw=pandas2ri.py2ri(pd.DataFrame(y))
        ro.globalenv['x_raw'] = x_raw
        ro.globalenv['y_raw'] = y_raw
        R('x<- model.matrix( ~ ., data = x_raw)')
        sumfit=R('sum_fit <- summary(coxph(formula=Surv(time=y_raw$Survival_in_days,event=as.numeric(y_raw$Status))~ ., data = as.data.frame(x)))')
        R('fit<-coxph(formula=Surv(time=y_raw$Survival_in_days,event=as.numeric(y_raw$Status))~ ., data = as.data.frame(x))')
        coeffs=sumfit.rx2('coefficients')
        confints=sumfit.rx2('conf.int')
        concordance=sumfit.rx2('concordance')
        df1=r_matrix_to_data_frame(coeffs)
        print(df1)
        df2=r_matrix_to_data_frame(confints)
        coef_conf=pd.concat([df1,df2], axis=1)
        print(R('as.data.frame(anova(fit))'))
        con=print ("Concordance :",concordance[0])
        conse=print ("Concordance standard error:",concordance[1])
        return con
        return conse
    

    def encoding(x):
        def r_matrix_to_data_frame(r_matrix):
            """
            Convert an R matrix into a Pandas DataFrame
            """
            from rpy2.robjects import pandas2ri
            array = pandas2ri.ri2py(r_matrix)
            return pd.DataFrame(array,index=r_matrix.names[0],columns=r_matrix.names[1])
        pandas2ri.activate()
        x_raw = pandas2ri.py2ri(x)
        ro.globalenv['x_raw'] = x_raw
        df =R('as.data.frame(model.matrix( ~ ., data = x_raw))')
        return df
   









    def fitcoxnet(x,y):
        """
        Fit a cox net model
        Input : data_x : Data frame in pandas with covariates 
                  y: Survival time array with time and censored event 
        Output : Coefficients selected 
        """
        pandas2ri.activate()
        x_raw = pandas2ri.py2ri(x)
        y_raw=pandas2ri.py2ri(pd.DataFrame(y))
        ro.globalenv['x_raw'] = x_raw
        ro.globalenv['y_raw'] = y_raw
        r_fct_string = """ 
        fitcoxnet <- function(DATAX,DATAY) {
        # Making Dummy Variables from the clinical Data
        x<- model.matrix( ~ ., data = DATAX)
        # Making a Survival Object 
        surv_obj <- Surv(time=DATAY$Survival_in_days,event=DATAY$Status)
        # create a glmnet cox object using lasso regularization and cross validation
        glmnet.cv <- cv.glmnet (x, surv_obj, family="cox")
        ## get the glmnet model on the full dataset
        glmnet.obj <- glmnet.cv$glmnet.fit
        # find lambda index for the models with least partial likelihood deviance
        optimal.lambda <- glmnet.cv$lambda.min
        lambda.index <- which(glmnet.obj$lambda==optimal.lambda)
        # take beta for optimal lambda 
        optimal.beta  <- glmnet.obj$beta[,lambda.index] 
        # find non zero beta coef 
        nonzero.coef <- abs(optimal.beta)>0 
        selectedBeta <- optimal.beta[nonzero.coef] 
        # take only covariates for which beta is not zero 
        selectedVar   <- x[,nonzero.coef]
        ## create a dataframe for trainSet with time, status and selected
        #variables in binary representation for evaluation in pec
        reformat_dataSet <- as.data.frame(cbind(surv_obj,selectedVar))
        # glmnet.cox only with meaningful features selected by stepwise
        #bidirectional AIC feature selection 
        glmnet.cox.meaningful <- step(coxph(Surv(time,status) ~
        .,data=reformat_dataSet),direction="both",trace=0)
        ## C-Index calculation 100 iter bootstrapping
        cIndexCoxglmnet<-c()
        features<-c()
        for (i in 1:100)
        {
                train <- sample(1:nrow(x), nrow(x), replace = TRUE) 
                reformat_trainSet <- reformat_dataSet [train,]
                glmnet.cox.meaningful.test <- step(coxph(Surv(time,status) ~
        .,data=reformat_trainSet),direction="both",trace=0)
                lbls<- as.vector(attr(glmnet.cox.meaningful.test$terms,"term.labels"))
                features<- c(features,lbls)
                varnames <- lapply(lbls, as.name)
                testdf<- as.data.frame(x)
                selectedVarCox   <-testdf[ names(testdf)[names(testdf) %in% varnames] ] 
                reformat_testSet <- as.data.frame(cbind(surv_obj,selectedVarCox))
                reformat_testSet <- reformat_dataSet [-train,]
            cIndexCoxglmnet <- c(cIndexCoxglmnet,
        1-rcorr.cens(predict(glmnet.cox.meaningful,
        reformat_testSet),Surv(reformat_testSet$time,reformat_testSet$status))[1])
        }
        cIndexm<- mean (unlist(cIndexCoxglmnet),rm.na=TRUE)
        cIndexstd<- sd(unlist(cIndexCoxglmnet))
        Finalresult=list(coefficients=glmnet.cox.meaningful,Average_Concordance=cIndexm,concordance_sd=cIndexstd,lambda=optimal.lambda)
        return(Finalresult)} """
        r_pkg = STAP(r_fct_string, "r_pkg")
        print(r_pkg.fitcoxnet(x_raw,y_raw))
    
    def fitcoxnet_nostep(x,y):
        """
        Fit a cox net model
        Input : data_x : Data frame in pandas with covariates 
                  y: Survival time array with time and censored event 
        Output : Coefficients selected 
        """
        pandas2ri.activate()
        x_raw = pandas2ri.py2ri(x)
        y_raw=pandas2ri.py2ri(pd.DataFrame(y))
        ro.globalenv['x_raw'] = x_raw
        ro.globalenv['y_raw'] = y_raw
        r_fct_string = """ 
        fitcoxnet <- function(DATAX,DATAY) {
        # Making Dummy Variables from the clinical Data
        x<- model.matrix( ~ ., data = DATAX)
        # Making a Survival Object 
        surv_obj <- Surv(time=DATAY$Survival_in_days,event=DATAY$Status)
        # create a glmnet cox object using lasso regularization and cross validation
        glmnet.cv <- cv.glmnet (x, surv_obj, family="cox")
        ## get the glmnet model on the full dataset
        glmnet.obj <- glmnet.cv$glmnet.fit
        # find lambda index for the models with least partial likelihood deviance
        optimal.lambda <- glmnet.cv$lambda.min
        lambda.index <- which(glmnet.obj$lambda==optimal.lambda)
        # take beta for optimal lambda 
        optimal.beta  <- glmnet.obj$beta[,lambda.index] 
        # find non zero beta coef 
        nonzero.coef <- abs(optimal.beta)>0 
        selectedBeta <- optimal.beta[nonzero.coef] 
        # take only covariates for which beta is not zero 
        selectedVar   <- x[,nonzero.coef]
        ## create a dataframe for trainSet with time, status and selected
        #variables in binary representation for evaluation in pec
        reformat_dataSet <- as.data.frame(cbind(surv_obj,selectedVar))
        # glmnet.cox only 
        glmnet.cox.meaningful <- coxph(Surv(time,status) ~
        .,data=reformat_dataSet)
        ## C-Index calculation 100 iter bootstrapping
        cIndexCoxglmnet<-c()
        features<-c()
        for (i in 1:100)
        {
                train <- sample(1:nrow(x), nrow(x), replace = TRUE) 
                reformat_trainSet <- reformat_dataSet [train,]
                glmnet.cox.meaningful.test <- coxph(Surv(time,status) ~
        .,data=reformat_trainSet)
                lbls<- as.vector(attr(glmnet.cox.meaningful.test$terms,"term.labels"))
                features<- c(features,lbls)
                varnames <- lapply(lbls, as.name)
                testdf<- as.data.frame(x)
                selectedVarCox   <-testdf[ names(testdf)[names(testdf) %in% varnames] ] 
                reformat_testSet <- as.data.frame(cbind(surv_obj,selectedVarCox))
                reformat_testSet <- reformat_dataSet [-train,]
            cIndexCoxglmnet <- c(cIndexCoxglmnet,
        1-rcorr.cens(predict(glmnet.cox.meaningful,
        reformat_testSet),Surv(reformat_testSet$time,reformat_testSet$status))[1])
        }
        cIndexm<- mean (unlist(cIndexCoxglmnet),rm.na=TRUE)
        cIndexstd<- sd(unlist(cIndexCoxglmnet))
        Finalresult=list(coefficients=glmnet.cox.meaningful,Average_Concordance=cIndexm,concordance_sd=cIndexstd)
        return(Finalresult)} """
        r_pkg = STAP(r_fct_string, "r_pkg")
        print(r_pkg.fitcoxnet(x_raw,y_raw))








    def fitrandomforest(x,y):
        """
        Fit a random forest model
        Input : data_x : Data frame in pandas with covariates 
                  y: Survival time array with time and censored event 
        Output : Importance values and concordance
        """
        def r_matrix_to_data_frame(r_matrix):
            """
            Convert an R matrix into a Pandas DataFrame
            """
            from rpy2.robjects import pandas2ri
            array = pandas2ri.ri2py(r_matrix)
            return pd.DataFrame(array,
                        index=r_matrix.names[0],
                        columns=r_matrix.names[1])
        pandas2ri.activate()
        x_raw = pandas2ri.py2ri(x)
        y_raw=pandas2ri.py2ri(pd.DataFrame(y))
        ro.globalenv['x_raw'] = x_raw
        ro.globalenv['y_raw'] = y_raw
        R('x<- model.matrix( ~ ., data = x_raw)')
        R('dform<- as.data.frame(x)')
        R('dform[1] <- NULL')
        R('ranger_model <- ranger(Surv(time=y_raw$Survival_in_days,event=as.numeric(y_raw$Status)) ~.,data=dform,num.trees = 500, importance = "permutation",seed = 1)')
        R('df1<- as.data.frame(importance_pvalues(ranger_model))')
        R('df2<-df1[order(df1$importance,decreasing = TRUE),]')
        #impnames =R('as.data.frame(row.names(df2))')
        imp=R('df1[order(df1$importance,decreasing = TRUE),]')
        #imp_pdf=pd.concat([impnames,imp], axis=1)
        craw=R('1-ranger_model$prediction.error')
        cindex=print ("Concordance :",craw)
        return imp
        return cindex
    
    def fitcoxboost(x,y):
        """
        Fit a Cox Boost Model
        Input : data_x : Data frame in pandas with covariates 
                  y: Survival time array with time and censored event 
        Output : Importance values and concordance
        """
        def r_matrix_to_data_frame(r_matrix):
            """
            Convert an R matrix into a Pandas DataFrame
            """
            from rpy2.robjects import pandas2ri
            array = pandas2ri.ri2py(r_matrix)
            return pd.DataFrame(array,
                        index=r_matrix.names[0],
                        columns=r_matrix.names[1])
        pandas2ri.activate()
        x_raw = pandas2ri.py2ri(x)
        y_raw=pandas2ri.py2ri(pd.DataFrame(y))
        ro.globalenv['x_raw'] = x_raw
        ro.globalenv['y_raw'] = y_raw
        R('x<- model.matrix( ~ ., data = x_raw)')
        R('censor<-as.integer(as.logical(y_raw$Status))')
        R('cv.res <- cv.CoxBoost(time=y_raw$Survival_in_days,status=censor,x=x,maxstepno=100,K=10,type="verweij",penalty=100)')
        R('optim.res <- optimCoxBoostPenalty(time=y_raw$Survival_in_days,status=censor,x=x,trace=TRUE,start.penalty=500)')
        R('fit <- CoxBoost(time=y_raw$Survival_in_days,status=censor,x=x,stepno=optim.res$cv.res$optimal.step,penalty=optim.res$penalty)')
        coefs=R('as.data.frame(coef(fit))')
        print(R('as.data.frame(estimPVal(fit,x))'))
        return coefs
        return pvals


