# include <iostream> 
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//   A C++ riff on q binomial option tree pricing model function I made during my time at the University of Toronto based   //
//   on material from  Derivative Markets and International Financial Management.  Apolgies in advanced,                    //
//                               this is my first ever C++ program.  Things might get messy.                                //
//                                      Copyright (c) 2025 Lunes Maibeche                                                   //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// TODO:
// 1. Implement the Binomial Option Pricing Model for European calls and American options.
// 2. Implement testing for the BOPM function
// 3. Implement automated data retrieval for option data (e.g., using an API to get real-time stock prices, interest rates, etc.)
// 4. Implement a user interface for the BOPM function (e.g., a command-line interface or a graphical user interface).
// 5. Learn how the how to leverage dynamic memory allocation (isn't that the whole point of C++?)


    // We're going to do some explaining here:
    // This approach is based off of three important optimizations noted by Desmond J Higham
    // when implementing the binomial option pricing model in MATLAB. 
    // 1. We can find the cutoff point past which the option payoff is zero in order to avoid
    //   calculating zero payoffs while also maintaining a smaller array for the terminal option values.
    // 2. We can use binomial expansion to calculate with minimal iteration
    // 3. We can use logarithms to avoid overflow/underflow issues.

double BOPEuroPut(double spot, double exercise, double rfRate, double timeToMaturity, double volatility, int steps) {
   // Intialize all the variables that we're going to be using 
    const double timeInterval = timeToMaturity/steps;                                                              // Time interval per step
    const double upFactor = std::exp(volatility * std::sqrt(timeInterval));                                        // Up factor
    const double downFactor = 1 / upFactor;                                                                        // Down factor
    const double discountFactor = std::exp(-rfRate * timeInterval);                                                // Discount factor per step
    const double upProbability = (std::exp(rfRate * timeInterval) - downFactor) / (upFactor - downFactor);         // Risk-neutral probability of up move
    const double downProbability = 1 - upProbability;                                                               // Risk-neutral probability of down move
    
    double rawIndex;
    // Now IN THEORY, this empty scope should be able to free up memory once the calculation is done WITHOUT 
    // messing up rawIndex in the process. But at the same time the compiler might be doing this anyways.
    {
        double tempVar1 = std::log((exercise * upFactor) / (spot * std::pow(downFactor, steps + 1)));
        double tempVar2 = std::log(upFactor / downFactor);
        rawIndex = tempVar1 / tempVar2;
    }
    // This is changes the previous float estimate to an integer index for the binomial tree.
    int cutoff = static_cast<int>(std::floor(rawIndex));
    // We gotta make sure that the cutoff is within the bounds of the binomial tree
    if (cutoff > (steps + 1)) {
        cutoff = steps + 1;
    }
    if (cutoff < 1){
        cutoff = 1;
    }
    
    // Find possible option value at maturity (based on the binomial tree)
    std::vector<double> payoffs(cutoff, 0.0);
    // I'm not sure if this dangerous because I haven't assigned z as a size
    for (int i = 0; i < cutoff; ++i) {
        int d_exp = steps - i;
        int u_exp = i;
        double term = spot * std::pow(downFactor, d_exp) * std::pow(upFactor, u_exp);
        payoffs[i] = exercise - term;
    }
   
    // Now we gotta initiate the binomial coefficient calculation vectors
    double tempVar1;
    tempVar1 = 0;
    double tempVar2;
    tempVar2 = 0;
    std::vector<double> upLogProbabilities(cutoff, 0.0);
    std::vector<double> downLogProbabilities(cutoff, 0.0);
    downLogProbabilities[0] = std::log(downProbability) * steps; // Down probability for 0 up moves

    // I am so proud of this loop - basically smooshed what took 4 vectors in MATLAB into one for loop
    for (int i = 1; i < cutoff; ++i) {
        double decreasingSeries = static_cast<double>(steps - i + 1);        // Decreasing series from steps to steps - z + 2
        double increasingSeries = static_cast<double>(i);                    // Increasing series from 1 to z-1
        tempVar1 += std::log(decreasingSeries);
        tempVar2 += std::log(increasingSeries);
        upLogProbabilities[i] = std::log(upProbability) * increasingSeries;
        downLogProbabilities[i] = std::log(downProbability) * (decreasingSeries-1);
    }

    double tmp1 = tempVar1 - tempVar2; // This is the log of the binomial coefficient
    double finalValue = 0.0;
    for(int i = 1; i < cutoff; ++i) {
        finalValue += payoffs[i] * std::exp(tmp1 + upLogProbabilities[i] + downLogProbabilities[i]);
    }
    finalValue = finalValue * discountFactor; // Discount the final value to present value
    return finalValue;
}



int main() {
    std::cout << BOPEuroPut(65.5, 68, 0.048, 1, 0.325, 3) << std::endl;
    // Aaaaaand its wrong. 
    // DEBUGGING TIME
    // I really should switch from VSCode to Cling, I thought I was built different
    return 0;
}
