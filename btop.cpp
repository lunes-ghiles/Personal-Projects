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


// Function header goes here

void BOPEuroPut(double spot, double exercise, double rfRate, double timeToMaturity, double volatility, int steps) {
   // Intialize all the variables that we're going to be using 
    const double timeInterval = timeToMaturity/steps;                                                              // Time interval per step
    const double upFactor = std::exp(volatility * std::sqrt(timeInterval));                                        // Up factor
    const double downFactor = 1 / upFactor;                                                                        // Down factor
    const double discountFactor = std::exp(-rfRate * timeInterval);                                                // Discount factor per step
    const double probability = (std::exp(rfRate * timeInterval) - downFactor) / (upFactor - downFactor);           // Risk-neutral probability of up move
   
    // We're going to do some explaining here:
    // This approach is based off of three important optimizations noted by Desmond J Higham
    // when implementing the binomial option pricing model in MATLAB. 
    // 1. We can find the cutoff point past which the option payoff is zero in order to avoid
    //   calculating zero payoffs while also maintaining a smaller array for the terminal option values.
    // 2. We can use vectorization and binomial expansion to calculate with minimal iteration
    // 3. We can use logarithms to avoid overflow/underflow issues.


    // This derives the aforementioned cutoff point, which is the point past which the option payoff becomes zero.
    double tempVar1 = std::log((exercise * upFactor) / (spot * std::pow(downFactor, steps + 1)));
    double tempVar2 = std::log(upFactor / downFactor);
    double rawIndex = tempVar1 / tempVar2;

    // This is changes the previous float estimate to an integer index for the binomial tree.
    int cutoff = static_cast<int>(std::floor(rawIndex));
    // We gotta make sure that the cutoff is within the bounds of the binomial tree
    if (cutoff < 0) {
        cutoff = steps + 1;
    }
    cutoff = std::min(steps + 1, cutoff);
    z = std::max(1, z);

    // Find possible option value at maturity (based on the binomial tree)
    std::vector<double> W(z);
    // I'm not sure if this dangerous because I haven't assigned z as a size
    for (int i = 0; i < z; ++i) {
        int d_exp = N - i;
        int u_exp = i;
        double term = S * std::pow(d, d_exp) * std::pow(u, u_exp);
        W[i] = X - term;
    }

    double tmp1 = std::accumulate(/* This is where we compute the binomial coefficient */);
}


int main() {
    // This is basically just a place holder for now so we can run debugger without the
   // compiler throwing a fit. SPOILER: It still throws a fit.
    std::cout << "BOPM calculation is not yet implemented." << '\n';
    return 0;
}
