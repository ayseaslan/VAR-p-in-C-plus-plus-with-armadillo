#include <iostream>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;

// function to read data, alternative to arma's load 
mat readCSV(const std::string& filename, const std::string& delimeter = ",")
{
    ifstream csv(filename);
    vector<vector<double>> datas;

    for (string line; getline(csv, line); ) {

        vector<double> data;

        // split string by delimeter
        auto start = 0U;
        auto end = line.find(delimeter);
        while (end != std::string::npos) {
            data.push_back(std::stod(line.substr(start, end - start)));
            start = end + delimeter.length();
            end = line.find(delimeter, start);
        }
        data.push_back(std::stod(line.substr(start, end)));
        datas.push_back(data);
    }

    mat data_mat = zeros<mat>(datas.size(), datas[0].size());

    for (int i = 0; i < datas.size(); i++) {
        mat r(datas[i]);
        data_mat.row(i) = r.t();
    }

    return data_mat;
}

// function to transform the input matrix data to timelagged input matrix X
void makeTimelagged(int p, int c, int y_c, mat& data, mat& X, mat& Y) {
    // MAKE THE TIMELAGGED INPUT MATRIX X AND PREDICTED VAR. Y
    for (int i = p; i < data.n_rows; i++) {
        rowvec r(c + 1, fill::ones);

        // the first element is one, to model the scalar variable 
        int a = 1;

        // the predicted variable at time t 
        Y(i - p, 0) = data(i, y_c);

        // add the past lags of the variables, info. from time steps before t 
        for (int k = 1; k <= p; k++) {
            for (int j = 0; j < data.n_cols; j++) {
                r(a) = data(i - k, j);
                a++;
            }

        }
        cout << i - p << endl;
        // add the row to the input matrix 
        X.row(i - p) = r;
        // print the row data for inspection, optional 
        for (int j = 0; j < c + 1; j++) {
            cout << X(i - p, j) << " ";
        }
        cout << Y(i - p, 0) << " ";
        cout << endl;
    }
}

int main()
{
    // Load your time series data into Armadillo matrices, we are using the data without the feature labels and date information 
    mat data = readCSV("DailyDelhiClimateTrain.csv");  // Each column represents a variable, each row is a time point
     
    // Define the lag order (p) for the VAR model
    int p = 2;

    // set the dimensions of the time-lagged matrix 
    int n = data.n_rows - p; // number of samples we can use (during the first p samples we won't not have enough data to make predictions)
    int c = p * data.n_cols; // number of columns of the matrix with time lagged varibales 
    int y_c = 0; // select the predicted variable, the column index 0,1,2 or 3
    mat X = zeros<mat>(n, c + 1); // +1 because of the additional scalar variable 
    mat Y = ones<mat>(n, 1); 

    makeTimelagged(p, c, y_c, data, X, Y);

    // solve the linear system X*coef=Y, to fit coeff 
    mat coef;
    bool success = solve(coef, X, Y);

    if (!success) {
        cout << "Model estimation failed." << endl;
        return 1;
    }

    // Print the estimated coefficients
    cout << "VAR Model Coefficients:" << endl;
    cout << coef << endl;

    // now progress the test data set 
    mat Testdata = readCSV("DailyDelhiClimateTest.csv");
    n = Testdata.n_rows-p;
    mat X_test = zeros<mat>(n, c + 1); // +1 because of the additional scalar variable 
    mat Y_test = ones<mat>(n, 1);
    makeTimelagged(p, c, y_c, Testdata, X_test, Y_test);

    // obtain the predicted values for the test samples and compare it to the true values Y_test
    mat Y_hat = ones<mat>(n, 1);
    double dif = 0.0;
    for (int i = p; i < Testdata.n_rows; i++) {
        vec forecast = X_test.row(i - p) * coef;
        Y_hat(i - p, 0) = forecast(0);
        dif += (Y_hat(i - p, 0) - Y_test(i - p, 0)) * (Y_hat(i - p, 0) - Y_test(i - p, 0));
    }

    // write the rmse 
    dif = dif / Y_test.n_rows;
    dif = sqrt(dif);
    cout << "rmse " << dif << endl;




    return 0;
}
