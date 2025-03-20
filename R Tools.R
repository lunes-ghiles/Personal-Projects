# Packages
library(quantmod)
library(dplyr)

setDefaults(getSymbols, src = 'yahoo')

# Function to initialize stock information, with either log or arithmetic
# returns
init_ticker = function(ticker, year, log.Returns, arth.Returns){
  getSymbols(ticker)
  df = data.frame(Cl(get(ticker)[year]))
  if(log.Returns == TRUE){
    logS = log(df[,1])
    u = (c(NA,diff(logS)))
    df$log.Returns = u
  }
  if(arth.Returns == TRUE){
    u = c(NA, diff(df[,1]) / head(df[,1], -1))
    df$arth.Returns = u
  }
  return(df)
}

# Example case: Initialize for JPM and SHOP
dfJPM = init_ticker("JPM", "2024", TRUE, TRUE)
dfSHOP = init_ticker("SHOP", "2024", TRUE, TRUE)

# One-Day 99% VaR Calculation using Variance-Covariance Approach Function
VaR = function(year, ticker, portfolio.weight = 1){
  df = init_ticker(ticker, year, TRUE, FALSE)
  sd = portfolio.weight * sd(df$log.Returns[-1])
  N = qnorm(0.99, mean = 0, sd = 1)
  return(sd * N)
}

# Let's do a slightly more sophisticated VaR calculation, where
# we estimate the multivariate normal distribution

# Correlation of the two assets
corr.coeff = cor(dfJPM$log.Returns[-1], dfSHOP$log.Returns[-1])

# Covariance of the two assets
cov.portfolio = cov(dfJPM$log.Returns[-1], dfSHOP$log.Returns[-1])

# Standard deviation of equally weighted portfolio
C = matrix(
  c(sd.SHOP**2, cov.portfolio, cov.portfolio, sd.JPM**2),
  nrow = 2,
  ncol = 2
)

alpha = matrix(c(0.5,0.5), nrow = 2, ncol = 1)

S2.ewp = (t(alpha) %*% C %*% alpha)[1,1]

sd.ewp = sqrt(S2.ewp)*10**6

VaR.ewp = sd.ewp * 2.33

# Alternative Approach
sdSHOP.ewp = sd.SHOP * 500000
sdJPM.ewp = sd.JPM * 500000

sd.ewp2 = sqrt(sdJPM.ewp^2 + sdSHOP.ewp^2 + 2*corr.coeff*sdSHOP.ewp*sdJPM.ewp)

# Confirming they are indeed equal
all.equal(sd.ewp,sd.ewp2)

# HISTORICAL SIMULATION TIME

# We have 252 days of historical data, with today being the 252nd day

# There will be 251 simulation trials

simulation = data.frame(
   seq(1,251),seq(1,251),seq(1,251), seq(1,251), seq(1,251), seq(1,251), seq(1,251)
   ,seq(1,251)
  )

colnames(simulation) = c(
  "SHOP", "JPM", "SHOP Portfolio Value", "JPM Portfolio Value", 
  "EWP Portfolio Value","SHOP P/L", "JPM P/L", "EWP P/L"
  )

simulation$SHOP = dfSHOP[252,1] * (tail(dfSHOP$SHOP.Close,-1)/head(dfSHOP$SHOP.Close,-1))
simulation$JPM = dfJPM[252,1] * (tail(dfJPM$JPM.Close,-1)/head(dfJPM$JPM.Close,-1))

# We're not taking into account the initial no. of stocks in the portfolio
init_SHOP = 10 ** 6 * 0.5/dfSHOP[252,1]
init_JPM = 10**6 * 0.5/dfJPM[252,1]

simulation$`SHOP Portfolio Value` = (simulation$SHOP * 10**6/dfSHOP[252,1])
simulation$`JPM Portfolio Value` = (simulation$JPM * 10**6/dfJPM[252,1]) 
simulation$`EWP Portfolio Value` = (simulation$SHOP * init_SHOP + simulation$JPM * init_JPM)
simulation$`SHOP P/L` = simulation$`SHOP Portfolio Value` - 10**6
simulation$`JPM P/L` = simulation$`JPM Portfolio Value` - 10**6
simulation$`EWP P/L` = simulation$`EWP Portfolio Value` - 10**6



SHOPsortedGL = sort(simulation$`SHOP P/L`, decreasing = TRUE)
JPMsortedGL = sort(simulation$`JPM P/L`, decreasing = TRUE)
EWPsortedGL = sort(simulation$`EWP P/L`, decreasing = TRUE)

CVaRSHOP = mean(tail(SHOPsortedGL, n=3))
CVaRJPM = mean(tail(JPMsortedGL, n=3))
CVaREWP = mean(tail(EWPsortedGL, n=3))

# Now let's repeat this on a bigger scale why don't we

getSymbols(c("^DJI", "^FTSE", "^FCHI", "^N225", "GBPUSD=X", "EURUSD=X", "JPYUSD=X"), from = "2023-01-01", to = "2024-12-31")

df2 = merge(Cl(DJI),Cl(FTSE),Cl(FCHI),Cl(N225), Cl(`GBPUSD=X`), Cl(`EURUSD=X`),
            Cl(`JPYUSD=X`), all = TRUE)

df2 = na.locf(df2)
df2 = na.omit(df2)


dija = df2$DJI.Close
ftse = df2$FTSE.Close
fchi = df2$FCHI.Close
n225 = df2$N225.Close

df2.adjusted = data.frame(dija, ftse * df2$GBPUSD.X.Close, fchi*df2$EURUSD.X.Close,
                          n225 * df2$JPYUSD.X.Close)
  
colnames(df2.adjusted) = c(
  "Dow Jones", "FTSE (adjusted)", "FCHI (adjusted)", "Nikkei 225 (adjusted)"
)


# We have 520 days of historical data, with today being the 520th day

# There will be 519 simulation trials, with some having NA values

simulation2 = data.frame(
  seq(1,580),seq(1,580),seq(1,580), seq(1,580), seq(1,580), seq(1,580)
)

colnames(simulation2) = c(
  "DJIA", "FTSE", "CAC 40", "Nikkei 225", "Portfolio Value (000s)", "Gain/Loss (000s)"
)

last_entries = tail(df2.adjusted, n=1)
init_dija = 5000/last_entries[[1,1]]
init_ftse = 2000/last_entries[[1,2]]
init_fchi = 4000/last_entries[[1,3]]
init_n225 = 1000/last_entries[[1,4]]


returns_djia = df2.adjusted$`Dow Jones` / lag(df2.adjusted$`Dow Jones`)  
returns_ftse = df2.adjusted$`FTSE (adjusted)` / lag(df2.adjusted$`FTSE (adjusted)`)  
returns_fchi = df2.adjusted$`FCHI (adjusted)`/lag(df2.adjusted$`FCHI (adjusted)`)
returns_n225 = df2.adjusted$`Nikkei 225 (adjusted)`/lag(df2.adjusted$`Nikkei 225 (adjusted)`)

# Remove the first NA value
returns_djia = returns_djia[-1]
returns_ftse = returns_ftse[-1]
returns_fchi = returns_fchi[-1]
returns_n225 = returns_n225[-1]

# Multiply by the last observed value of DJIA
simulation2$DJIA = as.numeric(last_entries[[1,1]]) * returns_djia
simulation2$FTSE = as.numeric(last_entries[[1,2]]) * returns_ftse
simulation2$`CAC 40` = as.numeric(last_entries[[1,3]]) * returns_fchi
simulation2$`Nikkei 225` = as.numeric(last_entries[[1,4]]) * returns_n225

# Remove all scenarios with NA values

# Fill the Portfolio Value column
simulation2$`Portfolio Value (000s)` = init_dija*simulation2$DJIA +
  init_ftse*simulation2$FTSE + init_fchi*simulation2$`CAC 40` +
  init_n225*simulation2$`Nikkei 225`

simulation2$`Gain/Loss (000s)` = simulation2$`Portfolio Value (000s)` - (5000+2000+4000+1000)

# 99\%, we want to find the 0.99 * 580th worst entry approx 575

PLsorted = sort(simulation2$`Gain/Loss (000s)`, decreasing = TRUE)
VaRsim2 = PLsorted[[574]]

simulation.weighted = simulation2

weights = (0.995^(580 - seq(1,580))*(1-0.995))/(1 - 0.995**(580))

simulation.weighted$Weights = weights

df_sorted <- simulation.weighted[order(simulation.weighted$`Gain/Loss (000s)`), 
                                 c("Gain/Loss (000s)", "Weights")]

df_sorted$`Cumulative Weights` = cumsum(df_sorted$Weights)


