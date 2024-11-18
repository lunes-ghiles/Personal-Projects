library(lubridate)
library(TSstudio)
library(quantmod)

# For now just using this to answer an assignment, will seek to 
# flesh it out, because for now, this can basically just be done in excel.

#TO DO LIST:
# Definitely want to incorporate quantmod to automatically scrape data []
# Statistical tests on the test statistics such as hypothesis testing,
# p-values, Confidence Intervals, Likelihood Ratio Test, ect []
# Streamline my vectorized operations - that is one ugly FOR loop []

# Log Returns is something I will certainly need in future in order to apply any more advanced financial modelling. Best 
# to make this as efficient as I can.
log_stats <- function(timeframe){
  stock_history = paste0("/Users/lunesm/Desktop/NVDA_", timeframe, ".csv")
  stockinfo <- read.csv(stock_history, header = TRUE, stringsAsFactors = TRUE)
  stockinfo$Date <- ymd(stockinfo$Date)
  sol_vector <- c()
  
  for (i in 1:(nrow(stockinfo)-1)){
    price1 <- as.numeric(stockinfo[i,2])
    price2 <- as.numeric(stockinfo[i+1,2])
    log_return <- log(price2/price1)
    sol_vector <- c(sol_vector,log_return)
    }
  average <- (mean(sol_vector))
  sample_sd <- (sd(sol_vector))

  if(timeframe == "DAILY"){
    tau <- 1/252
  }
  
  if(timeframe == "WEEKLY"){
    tau <- 1/52
  }
  
  if(timeframe == "MONTHLY"){
    tau <- 1/12
  }
  
  sample_vol <- sample_sd/(sqrt(tau))
  SE <- sample_sigma/(sqrt(2*length(sol_vector)))  
  
}



# a. Estimate the stock price volatility and the standard error using daily 
# data

log_stats("DAILY")

print(paste0("Stock price volatility is", sample_vol, "."))
      
print(paste0("Estimator standard error is", SE, "."))
      
# b. Estimate the stock price volatility and the standard error using weekly 
# data

log_stats("WEEKLY")

print(paste0("Stock price volatility is", sample_vol, "."))

print(paste0("Estimator standard error is", SE, "."))

# c. Estimate the stock price volatility and the standard error using monthly 
# data

log_stats("MONTHLY")

print(paste0("Stock price volatility is", sample_vol, "."))

print(paste0("Estimator standard error is", SE, "."))

