library(lubridate)
library(TSstudio)

average_logarithmic_return <- function(){
  stock_history = "/cmshome/maibeche/Desktop/PriceHistoryAMZN.csv"
  stockinfo <- read.csv(stock_history, header = TRUE, stringsAsFactors = TRUE)
  stockinfo <- stockinfo[-c(1),]
  stockinfo$Date <- ymd(stockinfo$Date)
  sol_vector <- c()
  
  for (i in 1:(nrow(stockinfo)-1)){
    date1 <- stockinfo[i,1]
    date2 <- stockinfo[i+1,1]
    price1 <- as.numeric(stockinfo[i,2])
    price2 <- as.numeric(stockinfo[i+1,2])
    
    if (difftime(date2, date1, units = "days") == 1){
    log_return <- log(price2/price1)
    sol_vector <- c(sol_vector,log_return)
    }
  }
  average <- (sum(sol_vector))
  print(average)
}


plot(stockinfo)