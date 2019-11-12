#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <map>
#include <omp.h>
#include <bits/stdc++.h>

using namespace std;

#define EMPTY -2.0
#define ll long long
#define MAT vector<vector<double> >

class CF {
private:
    MAT userToItem, itemToItem;
    vector<double> average;
    int itemSize, userSize;
public:
    CF(MAT userToItem) {
        this->userToItem = userToItem;
        
        userSize = (int)userToItem.size()-1;
        itemSize = (int)userToItem.back().size()-1;
        average = vector<double> (userSize+1, 0.0);
    }
    
    void run () {
        buildItemToItemC();
    }
    
    void getAverage() {
        #pragma omp parallel for schedule(dynamic) //chunk size is dependent on the machine
        for(int i=1;i<=userSize;i++) {
            int cnt = 0;
            int par_temp = 0;

            // used reduction because it contains the local copy of par_temp and update without over-write
            #pragma omp parallel for reduction(+:par_temp,cnt)
            for(int j=1;j<=itemSize;j++) {
                if(userToItem[i][j] == EMPTY)
                	continue;
                cnt++;
                par_temp+= userToItem[i][j]; // to reduce over-write condition

            }

            average[i]=par_temp;
            average[i] /= (double)cnt;
        }
    }
    
    void buildItemToItemC() {
        itemToItem = vector<vector<double> > (itemSize+1, vector<double> (itemSize+1));
        
        #pragma omp parallel for schedule(static)
        for(int i=1; i<= itemSize; i++) {
            
            #pragma omp parallel for schedule(static)
            for(int j=1; j <= itemSize; j++) {
                double top = 0, bleft =0, bright = 0;
                
                int cnt = 0;

                #pragma omp parallel for reduction(+:top,bleft,bright)
                for(int k=1; k<= userSize; k++) {
                    if(userToItem[k][i] == EMPTY) continue;
                    if(userToItem[k][j] == EMPTY) continue;
            
                    cnt+=1;
                    top += userToItem[k][i]*userToItem[k][j];
                    
                    bleft += userToItem[k][i]*userToItem[k][i];
                    bright += userToItem[k][j]*userToItem[k][j];
                }
                
                if(cnt < 1) {
                    itemToItem[i][j] = 0;
                } else {
                    itemToItem[i][j] = top/(sqrt(bleft)*sqrt(bright));
                }
            }
        }
    }
    double get_mag(vector<double> v){
        double res=0;
        for(int i:v)res+=i*i;
        return sqrt(res);
    }
    double get_tri_similarity(vector<double> a,vector<double> b){
        std::vector<double> v;
        for(int i=0;i<a.size();++i){
            if(a[i]==EMPTY || b[i]==EMPTY)v.push_back(0);
            else v.push_back(a[i]-b[i]);
        }
        return 1 - get_mag(v)/(get_mag(a)+get_mag(b));
    }
    double get_jac_similarity(vector<double> a,vector<double> b){
        double res =0;
        for(int i=0;i<a.size();++i){
            if(a[i]==EMPTY || b[i]==EMPTY)res+=0.0;
            else res+=a[i]*b[i];
        }
        return res / (get_mag(a)+get_mag(b)-res);
    }

    double get_hybrid_similarity(vector<double> a,vector<double>b){
        return 0.5 * get_jac_similarity(a,b)*(get_tri_similarity(a,b)+1);
    }
    void buildItemToItemP() {
        getAverage();
        itemToItem = vector<vector<double> > (itemSize+1, vector<double> (itemSize+1));
        #pragma omp parallel for schedule(dynamic)
        for(int i=1; i<= itemSize; i++) {
            #pragma omp parallel for schedule(dynamic)
            for(int j=1; j <= itemSize; j++) {
                double top = 0, bleft =0, bright = 0;
                int cnt = 0;
                vector<double> a,b;
                for(int k=1; k<= userSize; k++) {
                    a.push_back(userToItem[k][i]);
                    b.push_back(userToItem[k][j]);
                    cnt+=1;
                }
                if(cnt < 1) {
                    itemToItem[i][j] = 0;
                } else {
                    itemToItem[i][j] = get_hybrid_similarity(a,b);
                }
            }
        }
    }
    
    int predict(int user, int item) {
        double top = 0;
        double bottom = 0;
        
        // given untrained items
        if(item > itemSize) {
            return 3;
        }
        
        #pragma omp parallel for reduction(+:bottom,top) schedule(auto)
        for(int i=1;i<=itemSize; i++) {
            if(userToItem[user][i] == EMPTY) continue;
            
            bottom += abs(itemToItem[item][i]);
            top += itemToItem[item][i]*userToItem[user][i];
        }
        
        if(bottom == 0) {
            return 3;
        }
        
        double rating = top/bottom;
        // return rating;
        return rating+0.5;
    }

    void get_mats(){
    	freopen("usertoitem.txt","w",stdout);
    	cout << "UserToItem Matrix:";
    	for(int i=1;i<=userSize;++i){
    		for(int j=1;j<=itemSize;++j){
    			cout << max(0.0,userToItem[i][j]) << ' ';
    		}
    		cout << endl;
    	}
    	freopen("itemtoitem.txt","w",stdout);
    	cout << "ItemToItem Matrix:\n";
    	for(int i=1;i<=itemSize;++i){
    		for(int j=1;j<=userSize;++j)
    			cout << itemToItem[j][i] << ' ';
    		cout << endl;
    	}

    	return;
    }
};

struct inputTuple{
    int user, item, rating;
};

class InputReader {
private:
    ifstream fin;
    MAT userToItem;
    vector<inputTuple> input;
public:
    InputReader(string filename) {
        fin.open(filename);
        if(!fin) {
            cout << filename << " file could not be opened\n";
            exit(0);
        }
        parse();
    }
    
    void parse() {
        int maxUser = 0, maxItem = 0;
        
        int user, item, rating, timeStamp;
        while(!fin.eof()) {
            fin >> user >> item >> rating >> timeStamp;
            
            input.push_back({user, item, rating});
            
            maxUser = max(maxUser, user);
            maxItem = max(maxItem, item);
        }
        input.pop_back();
        
        userToItem = vector<vector<double> > (maxUser+1, vector<double> (maxItem+1, EMPTY));

        #pragma  omp parallel for private(user,item,rating)
        for(int i=0;i<input.size();i++) {
            user = input[i].user, item = input[i].item, rating = input[i].rating;
            userToItem[user][item]= rating;
        }
    }
    
    vector<inputTuple> getInput() {
        return input;
    }
    
    MAT getUserToItem() {
        return userToItem;
    }
};

class OutputPrinter {
private:
    ofstream fout;
    string filename;
public:
    OutputPrinter(string filename) {
        this->filename = filename;
        fout.open(filename);
        if(!fout) {
            cout << filename << " file could not be writed\n";
            exit(0);
        }
    }
    void addLine(int user, int item, double rating) {
        fout << user << "\t" << item << "\t" << rating << endl;
    }
};

int main(int argc, const char * argv[]) {
	long time1 = omp_get_wtime();
    if(argc!=3) {
        cout << "Please follow this format. recommender.exe [base file name] [test file name]";
        return 0;
    }
    double test1 = 1.23;
    string baseFileName(argv[1]);
    string testFileName(argv[2]);
    
    InputReader baseInputReader(baseFileName);
    
    MAT userToItem = baseInputReader.getUserToItem();
    CF cf(userToItem);
    cf.run();
    
    InputReader testInputReader(testFileName);
    vector<inputTuple> test = testInputReader.getInput();
    
    OutputPrinter outputPrinter(testFileName.substr(0, 2)+".base_prediction.txt");
    
    double MAE = 0.0;
    

    #pragma omp parallel for reduction(+:MAE) 
    for(int i=0;i<test.size();i++) {
        int user = test[i].user;
        int item = test[i].item;
        int rating = test[i].rating;
        
        double predict = abs(cf.predict(user, item));
        
        MAE += abs(predict-rating);
        #pragma omp critical
        outputPrinter.addLine(user, item, predict);
    }
    double div = test1*(double)test.size();
    MAE /= div;
    cout << "MAE :" << MAE << endl;
    // cout << "Total time : " << omp_get_wtime()-time1 << "\n";

    // cf.get_mats();
    return 0;
}
