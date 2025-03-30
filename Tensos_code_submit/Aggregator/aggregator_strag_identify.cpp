#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <fstream>
#include <math.h>
#include "typedefs.h"
#include "socket.h"
#include "connection.h"
#include <thread>
#include <algorithm>
#include <mutex>
#include <iomanip>

mutex mtx;
using namespace std;


char address[] = "127.0.0.1";
const uint32_t MAX_WORKER_NUMBER = 32;
uint32_t TREE_NUMBER = 1;
uint32_t WORKER_NUMBER;
const uint32_t FIELD_SIZE = 1024;
const double my_lambda=0.05;
uint32_t QUANTILE_NUMBER;
uint32_t LAYER_NUMBER;
string out_file;
const uint32_t BLOCK = 10;
uint32_t FEATURE_NUMBER;
uint16_t port_cnt;
int socket_reconnect_cnt=0;
int server_in_bandwidth;
uint32_t NON_BLOCKING_OPTIM=1;
uint32_t PRE_ELECTION_OPTIM=1;
int pipeline_feature_batch=150;
int exp_all_correct=0;
const double my_EPS=1e-3;
double sample_reduce_coef=0.1;
double min_step2_feat_frac=0.2;
string middle_result_path="";
vector<int> total_straggler_num;
struct Instance
{
    string name;
    double label;
    vector<double> value;
    Instance () {}
};

union Binary_Float
{
    float value;
    char s[4];
}binary_float;

union Binary_Uint32
{
    uint32_t value;
    char s[4];
}binary_uint;

union Binary_Double
{
    double value;
    char s[8];
}binary_double;

string f2b(float x)
{
    string str;
    Binary_Float my_b_f;
    my_b_f.value = x;
    for (uint32_t i = 0; i < 4; i++)
        str += my_b_f.s[i];
    return str;
}

string d2b(double x)
{
    string str;
    Binary_Double my_b_w;
    my_b_w.value = x;
    for (uint32_t i = 0; i < 8; i++)
        str += my_b_w.s[i];
    return str;
}

string u2b(uint32_t x)
{
    string str;
    Binary_Uint32 my_b_u;
    my_b_u.value = x;
    for (uint32_t i = 0; i < 4; i++)
        str += my_b_u.s[i];
    return str;
}

float b2f(basic_string<char> str)
{
    Binary_Float my_b_f;
    for (uint32_t i = 0; i < 4; i++)
        my_b_f.s[i] = str[i];
    return my_b_f.value;
}

double b2d(basic_string<char> str)
{
    Binary_Double my_b_w;
    for (uint32_t i = 0; i < 8; i++)
        my_b_w.s[i] = str[i];
    return my_b_w.value;
}

uint32_t b2u(basic_string<char> str)
{
    Binary_Uint32 my_b_u;
    for (uint32_t i = 0; i < 4; i++)
        my_b_u.s[i] = str[i];
    return my_b_u.value;
}

void myspilt(string s, vector<string> &res, char ch = ',')
{
    if (s.empty()) return;
    string tmp;
    for (const auto &p : s)
    {
        if (p == ch)
        {
            if (!tmp.empty()) res.push_back(tmp);
            tmp.clear();
        }
        else tmp += p;
    }
    if (!tmp.empty()) res.push_back(tmp);
    return;
}


vector<string> read_line(ifstream &in, char ch=',')
{
    string s;
    getline(in, s);
    if (!s.empty() && s.back() == '\n') s.pop_back();
    vector<std::string> res;
    myspilt(s, res);
    return res;
}

struct Sock
{
    uint16_t port_out, port_in;
    CSocket sock_out, sock_in;
    uint32_t send_cnt = 0, recv_cnt = 0;
    void init()
    {
    }
    void link(char* dest_addr,char* self_addr,uint16_t myoutport, uint16_t myinport)
    {
        port_out=myoutport;
        port_in=myinport;
        connect(dest_addr, port_out, sock_out);
        listen(self_addr, port_in, &sock_in, 1);
    }
    void close()
    {
        sock_out.Close();
        sock_in.Close();
    }
    void send_specify(string s, int send_len){
        sock_out.Send(s.c_str(),send_len+1);
    }
    void send_cstr(const char *s)
    {
        assert(strlen(s) < FIELD_SIZE);
        send_cnt++;
        sock_out.Send(s, FIELD_SIZE);
    }
    void send_str(string s)
    {
        send_cstr(s.c_str());
    }
    void send_uint32(uint32_t x)
    {
        send_str(u2b(x));
    }
    void send_double(double x)
    {
        send_str(d2b(x));
    }
   

    string recieve_specify(int my_len){
        char s[my_len+1];
        sock_in.Receive(s, my_len+1);
        string str;
        for (uint32_t i = 0; i < my_len; i++)
            str += s[i];
        return str;
    }

    void recieve_cstr(char *s)
    {
        recv_cnt++;
        sock_in.Receive(s, FIELD_SIZE);
    }
    double recieve_double()
    {
        return b2d(recieve_str(8));
    }
    uint32_t recieve_uint()
    {
        return b2u(recieve_str(4));
    }
    string recieve_str(uint32_t len)
    {
        char s[FIELD_SIZE];
        memset(s, 0, sizeof s);
        recieve_cstr(s);
        string str;
        for (uint32_t i = 0; i < len; i++)
            str += s[i];
        return str;
    }
    
}sock[MAX_WORKER_NUMBER];




vector<string> receive_buffer;
int batch_receive_signal;
vector<double> worker_orig_comp_time;
vector<double> worker_step1_frac;
vector<double> need_feat_num_ahead;
vector<int> worker_sample_level;
double step2_feat_frac;
vector<vector<vector<Binary_Double>>> agg_histogram_g;
vector<vector<vector<Binary_Double>>> agg_histogram_h;


vector<vector<double>> quantiles;
int write_agg_hist_sig=0;
vector<int> finish_batch_num;
vector<vector<int>> pre_ele_whole_history;
vector<vector<int>> schedule_history;
vector<int> tmp_vec;
double cur_cdf_val_with_this_featfrac;
double benefit_per_tree[40];


int straggler_combo_idx=0;
vector<double> straggler_combo_benefit;
vector<vector<int> > straggler_combo_sample_level;
vector<double> straggler_step2_feat_frac;
int straggler_phase=0;

void split_finding(vector<Binary_Double> G, vector<Binary_Double> H, double lambda, uint32_t &split_point, double &max_score){
        assert(G.size() == H.size());
        double Gl = 0, Hl = 0;
        double Gsum = 0;
        for (const auto &p : G) Gsum += p.value;
        double Hsum = 0;
        for (const auto &p : H) Hsum += p.value;
        uint32_t candidate_split_point = G.size()-1;

        double current_max_score = 0.0;
        for (uint32_t i = 0; i < G.size(); i++)
        {
            Gl += G[i].value;
            Hl += H[i].value;
            double Gr = Gsum - Gl;
            double Hr = Hsum - Hl;
            //double tmp_score = Gl * Gl / (Hl + lambda) + Gr * Gr / (Hr + lambda) + Gsum * Gsum / (Hsum + lambda);
            double tmp_score = Gl * Gl / (Hl + lambda) + Gr * Gr / (Hr + lambda) - Gsum * Gsum / (Hsum + lambda);
            if (tmp_score > current_max_score)
            {
                candidate_split_point = i;
                current_max_score = tmp_score;
            }
        }

        //assert(candidate_split_point<G.size()-1);
        split_point = candidate_split_point;
        max_score = current_max_score;
        
        return;
}


void compute_need_feat_num_ahead(){
    double longest_time_for_step1=0;
    for(int i_worker=0;i_worker<WORKER_NUMBER;i_worker++){
        if(worker_orig_comp_time[i_worker]*worker_step1_frac[i_worker]>longest_time_for_step1)  
            longest_time_for_step1=worker_orig_comp_time[i_worker]*worker_step1_frac[i_worker];
    }//for
    for(int i_worker=0;i_worker<WORKER_NUMBER;i_worker++){
        double remain_to_comp_time=(worker_orig_comp_time[i_worker]*(1.0-worker_step1_frac[i_worker]))*step2_feat_frac;
        double wait_for_step1_complete_time=longest_time_for_step1-worker_orig_comp_time[i_worker]*worker_step1_frac[i_worker];
        if(remain_to_comp_time==0)  need_feat_num_ahead[i_worker]=0;
        else    need_feat_num_ahead[i_worker]=(wait_for_step1_complete_time/remain_to_comp_time)*FEATURE_NUMBER;
       
    }//for
}



void init_scheduler(){
    worker_orig_comp_time.resize(WORKER_NUMBER);
    worker_step1_frac.resize(WORKER_NUMBER);
    worker_step1_frac[0]=1.0;
    worker_step1_frac[1]=1.0;
    worker_step1_frac[2]=1.0;
    worker_step1_frac[3]=1.0;
    worker_step1_frac[4]=1.0;
    worker_step1_frac[5]=1.0;
    worker_step1_frac[6]=1.0;
    worker_step1_frac[7]=1.0;

    vector<int> straggler_temp_level;
    for (int i =0;i<WORKER_NUMBER;i++){
        straggler_temp_level.push_back(0);
    }
    straggler_combo_sample_level.push_back(straggler_temp_level);

    schedule_history.clear();

    need_feat_num_ahead.resize(WORKER_NUMBER);

    compute_need_feat_num_ahead();


}




double compute_cur_benefit(double feat_frac_x, double cdf_val_y){

    double orig_longest_time=0;
    for(int i=0;i<WORKER_NUMBER;i++){
        if(worker_orig_comp_time[i]>orig_longest_time)  orig_longest_time=worker_orig_comp_time[i];
    }//for
    double now_longest_time=0;
    for(int i=0;i<WORKER_NUMBER;i++){
        double cur_worker_time=worker_orig_comp_time[i]*worker_step1_frac[i]+worker_orig_comp_time[i]*(1-worker_step1_frac[i])*feat_frac_x;
        if(cur_worker_time>now_longest_time)    now_longest_time=cur_worker_time;
    }//for


    double my_gain=0;
    double my_loss=0;
    double all_nodes_correct_prob[LAYER_NUMBER-1];
    double my_score=0;
    for(int cur_layer=1;cur_layer<=LAYER_NUMBER-2;cur_layer++){
        int node_num=int(pow(2.0,cur_layer-1));
        all_nodes_correct_prob[cur_layer]=pow(cdf_val_y,node_num);
        double fail_prob=1.0-all_nodes_correct_prob[cur_layer];
        double prev_layer_all_correct=1.0;
        for(int i=1;i<cur_layer;i++)    prev_layer_all_correct=prev_layer_all_correct*all_nodes_correct_prob[i];
        my_score+= prev_layer_all_correct*((orig_longest_time-now_longest_time) - (now_longest_time*(LAYER_NUMBER-2-cur_layer)+orig_longest_time)*fail_prob);
        my_gain += prev_layer_all_correct * (orig_longest_time-now_longest_time);
        my_loss += prev_layer_all_correct * (now_longest_time*(LAYER_NUMBER-2-cur_layer)+orig_longest_time)*fail_prob;
    }//for

    return my_gain-my_loss;
}







void locate_worker_to_increase_sample_level(double max_gain_feat_frac,vector<int>& worker_list){
    int straggler_number=0;
    for(int i=0;i<WORKER_NUMBER;i++){
        if(straggler_combo_sample_level[straggler_combo_idx][i]>0)  straggler_number++;
    }//for
    if(straggler_number==WORKER_NUMBER-1) return;
    if(worker_orig_comp_time[0]>worker_orig_comp_time[WORKER_NUMBER-1]){
        worker_list.push_back(straggler_number);
    }//if
    else{
        worker_list.push_back(WORKER_NUMBER-1-straggler_number);
    }


}



void update_schedule_per_tree_adaptive_fast(int to_construct_tree_id){
    vector<int> straggler_temp_level;
    for(int i=0;i<WORKER_NUMBER;i++){
        worker_step1_frac[i]=pow(sample_reduce_coef,straggler_combo_sample_level[straggler_combo_idx][i]);
    }//for

    if(to_construct_tree_id==1){
        straggler_combo_benefit.push_back(0);

        straggler_step2_feat_frac.push_back(1.0);
        step2_feat_frac=straggler_step2_feat_frac[straggler_combo_idx];

        for(int i=0;i<WORKER_NUMBER;i++){
            worker_step1_frac[i]=pow(sample_reduce_coef,straggler_combo_sample_level[straggler_combo_idx][i]);
        }//for

        compute_need_feat_num_ahead();

        int cur_straggler_num=0;
        for(int i=0;i<WORKER_NUMBER;i++)
            if(worker_step1_frac[i]<0.99) cur_straggler_num++;
        total_straggler_num.push_back(cur_straggler_num);
        return;
    }//if 


    if(to_construct_tree_id==2){
        int max_time_w_id=0;
        for(int i_worker=0;i_worker<WORKER_NUMBER;i_worker++){
            if(worker_orig_comp_time[i_worker]>=worker_orig_comp_time[max_time_w_id]){
                max_time_w_id=i_worker;
            }//if
        }//for
        for (int i =0;i<WORKER_NUMBER;i++){
            if(i==max_time_w_id)   straggler_temp_level.push_back(1); 
            else    straggler_temp_level.push_back(0);
        }
        straggler_combo_sample_level.push_back(straggler_temp_level);
        straggler_combo_idx++;
        step2_feat_frac=min_step2_feat_frac;

        for(int i=0;i<WORKER_NUMBER;i++){
            worker_step1_frac[i]=pow(sample_reduce_coef,straggler_combo_sample_level[straggler_combo_idx][i]);
        }//for

        compute_need_feat_num_ahead();

        int cur_straggler_num=0;
        for(int i=0;i<WORKER_NUMBER;i++)
            if(worker_step1_frac[i]<0.99) cur_straggler_num++;
        total_straggler_num.push_back(cur_straggler_num);
        return;
    }//if 

    vector<int> true_feat_drift_distance;
    ifstream if_topN_cdf;
    if_topN_cdf.open(middle_result_path+"t10l8_topN_cdf.txt",ios::in);
    assert(if_topN_cdf.is_open());
    string line;
    int pre_ele_nodes_per_tree=int(pow(2.0,LAYER_NUMBER-2)-1);
    for(int i=0;i<to_construct_tree_id-2;i++){
        for(int node_num=0;node_num<pre_ele_nodes_per_tree;node_num++)
            getline(if_topN_cdf,line);
    }//for
    istringstream sin;
    for(int node_num=0;node_num<pre_ele_nodes_per_tree;node_num++){
        getline(if_topN_cdf,line);
        sin.clear();
        sin.str(line);
        string word;
        while(getline(sin,word,',')){
            true_feat_drift_distance.push_back(stoi(word));
        }//while
    }//for
    sort(true_feat_drift_distance.begin(),true_feat_drift_distance.end());


    double min_feat_frac_for_search=min_step2_feat_frac;
    double max_feat_frac_for_search=double(true_feat_drift_distance[true_feat_drift_distance.size()-1])/double(FEATURE_NUMBER);
    double leap_delta=(max_feat_frac_for_search-min_feat_frac_for_search)/10.0;
    double max_benefit=-1000000000;
    double max_gain_feat_frac=-1;
    double max_gain_cdf_val=-1;
    double cur_benefit;
    if(leap_delta<=0){
        max_benefit=compute_cur_benefit(min_step2_feat_frac,1.0);
        max_gain_feat_frac=min_step2_feat_frac;
        max_gain_cdf_val=1.0;
    }//if
    else{
        for(int i=0;i<10;i++){
            int my_idx=lower_bound(true_feat_drift_distance.begin(),true_feat_drift_distance.end(),FEATURE_NUMBER*(min_feat_frac_for_search+i*leap_delta))
                            -true_feat_drift_distance.begin();
            if(my_idx>=true_feat_drift_distance.size()) my_idx=true_feat_drift_distance.size()-1;
            double feat_frac_x=min_feat_frac_for_search+i*leap_delta;
            double cdf_val_y=double(my_idx)/double(true_feat_drift_distance.size());
            cur_benefit=compute_cur_benefit(feat_frac_x,cdf_val_y);
            if(cur_benefit>max_benefit){
                max_benefit=cur_benefit;
                max_gain_feat_frac=feat_frac_x;
                max_gain_cdf_val=cdf_val_y;
            }//if
        }//for
    }//else


    if(straggler_phase==0){
        straggler_combo_benefit.push_back(max_benefit);
        straggler_step2_feat_frac.push_back(max_gain_feat_frac);



        double total_max_benefit=0;
        int total_max_straggler_combo_idx;
        for(int i=0;i<=straggler_combo_idx;i++){
            if(straggler_combo_benefit[i]>total_max_benefit){
                total_max_benefit=straggler_combo_benefit[i];
                total_max_straggler_combo_idx=i;
            }//if
        }//for

        vector<int> worker_inc_list;
        worker_inc_list.clear();
        locate_worker_to_increase_sample_level(max_gain_feat_frac,worker_inc_list);
        if(((total_max_benefit-max_benefit)>0.1*total_max_benefit)||worker_inc_list.size()==0){
            straggler_phase=1;

            straggler_combo_idx=total_max_straggler_combo_idx;
            step2_feat_frac=straggler_step2_feat_frac[straggler_combo_idx];
        }
        else{
            step2_feat_frac=max_gain_feat_frac;

            for (int i =0;i<WORKER_NUMBER;i++){
                straggler_temp_level.push_back(straggler_combo_sample_level[straggler_combo_idx][i]);
            }//for
            for(int i=0;i<worker_inc_list.size();i++){
                straggler_temp_level[worker_inc_list[i]]=1;
            }//for
            straggler_combo_sample_level.push_back(straggler_temp_level);
            straggler_combo_idx++;
        }//else
    }//if
    else{
        straggler_combo_benefit[straggler_combo_idx]=(max_benefit+straggler_combo_benefit[straggler_combo_idx])/2.0;
        straggler_step2_feat_frac[straggler_combo_idx]=max_gain_feat_frac;
    

        double total_max_benefit=0;
        int total_max_straggler_combo_idx;
        for(int i=0;i<straggler_combo_benefit.size();i++){
            if(straggler_combo_benefit[i]>total_max_benefit){
                total_max_benefit=straggler_combo_benefit[i];
                total_max_straggler_combo_idx=i;
            }//if
        }//for
        straggler_combo_idx=total_max_straggler_combo_idx;
        step2_feat_frac=straggler_step2_feat_frac[straggler_combo_idx];
    }//else

    for(int i=0;i<WORKER_NUMBER;i++){
        worker_step1_frac[i]=pow(sample_reduce_coef,straggler_combo_sample_level[straggler_combo_idx][i]);
    }//for
    compute_need_feat_num_ahead();

    int cur_straggler_num=0;
    for(int i=0;i<WORKER_NUMBER;i++)
        if(worker_step1_frac[i]<0.99) cur_straggler_num++;
    total_straggler_num.push_back(cur_straggler_num);

    return;
}





void reset_scheduler_for_redo(){
    for (int i =0;i<WORKER_NUMBER;i++){
        worker_step1_frac[i]=1.0;
        need_feat_num_ahead[i]=0;
    }
}

void thread_receive_hist(int node_num_startID_curl,int worker_id){
    int remain_to_receive=2*8*node_num_startID_curl*FEATURE_NUMBER*QUANTILE_NUMBER;
    while(remain_to_receive>0){
        if(remain_to_receive>4194304){
            receive_buffer[worker_id]+=(sock[worker_id].recieve_specify(4194304));
            remain_to_receive-=4194304;
        }//if
        else{
            receive_buffer[worker_id]+=(sock[worker_id].recieve_specify(remain_to_receive));
            remain_to_receive=0;
        }//else
    }//while
}


int cmp(const pair<int, double>& x, const pair<int, double>& y)  
{  
    if(fabs(x.second-y.second)<my_EPS)
        return x.first<y.first;
    return x.second>(y.second+my_EPS);
} 

int cmp_id(const pair<int, double>& x, const pair<int, double>& y)  
{  
    
    return x.first<y.first;
} 

void find_topN_candidate(int start_feat_batch,int end_feat_batch,int topN,int *return_topN_feature,int node_num_startID_curl,
                        int record_whole_history=0){
    vector<pair<double, double>> candidate_split(FEATURE_NUMBER);
    vector<pair<int, double>> per_node_all_ranks(FEATURE_NUMBER);
    
    int pointer_count=0;
    for(uint32_t i_node = 0; i_node < node_num_startID_curl; i_node++){
        per_node_all_ranks.clear();
        int j_start=(start_feat_batch-1)*pipeline_feature_batch;
        int j_end=end_feat_batch*pipeline_feature_batch;
        if(j_end>FEATURE_NUMBER) j_end=FEATURE_NUMBER;
        for (uint32_t j = j_start; j < j_end; j++){
            uint32_t cur;
            split_finding(agg_histogram_g[i_node][j], agg_histogram_h[i_node][j], my_lambda, cur, candidate_split[j].second);
            candidate_split[j].first = cur;
            per_node_all_ranks.push_back(make_pair(j,candidate_split[j].second));
        }//for 
        sort(per_node_all_ranks.begin(),per_node_all_ranks.end(),cmp);
        if(record_whole_history){
            for(int j=0;j<FEATURE_NUMBER;j++){
                pre_ele_whole_history[i_node+node_num_startID_curl].push_back(per_node_all_ranks[j].first);
            }//for        
        }//if

        sort(per_node_all_ranks.begin(), per_node_all_ranks.begin()+topN, cmp_id);
        for(int i=0;i<topN;i++){
            return_topN_feature[pointer_count]=per_node_all_ranks[i].first;
            pointer_count++;
        }//for

    }//for 
}//void



void aggregate_histogram_perworker(int worker_id,int batch_id,int node_num_startID_curl){
    double *db_rcv_array;
    db_rcv_array=(double*)(receive_buffer[worker_id].c_str());
    int j_start=batch_id*pipeline_feature_batch;
    int j_end=(batch_id+1)*pipeline_feature_batch;
    if(j_end>FEATURE_NUMBER) j_end=FEATURE_NUMBER;
    while(1){
        if(write_agg_hist_sig==0){
            write_agg_hist_sig=1;
            break;
        }//if
    }//while
    for (uint32_t j = j_start; j < j_end; j++){
        for(uint32_t i_node = 0; i_node < node_num_startID_curl; i_node++){
                for (uint32_t q = 0; q < QUANTILE_NUMBER; q++){
                    agg_histogram_g[i_node][j][q].value+=*(db_rcv_array+2*((j-j_start)*node_num_startID_curl*QUANTILE_NUMBER+i_node*QUANTILE_NUMBER+q));
                    agg_histogram_h[i_node][j][q].value+=*(db_rcv_array+2*((j-j_start)*node_num_startID_curl*QUANTILE_NUMBER+i_node*QUANTILE_NUMBER+q)+1);
                }//for
        }//for
    }//for
    write_agg_hist_sig=0;
}

void thread_receive_hist_pre_ele(int node_num_startID_curl,int worker_id){
    for(int batch_id=0;batch_id<(FEATURE_NUMBER-1)/pipeline_feature_batch+1;batch_id++){
        string().swap(receive_buffer[worker_id]);
        receive_buffer[worker_id]="";
        
        int remain_to_receive=2*8*node_num_startID_curl*pipeline_feature_batch*QUANTILE_NUMBER;
        if(batch_id==(FEATURE_NUMBER-1)/pipeline_feature_batch+1-1)
            if(FEATURE_NUMBER%pipeline_feature_batch!=0)
                remain_to_receive=2*8*node_num_startID_curl*(FEATURE_NUMBER%pipeline_feature_batch)*QUANTILE_NUMBER;
        while(remain_to_receive>0){
            if(remain_to_receive>4194304){
                receive_buffer[worker_id]+=(sock[worker_id].recieve_specify(4194304));
                remain_to_receive-=4194304;
            }//if
            else{
                receive_buffer[worker_id]+=(sock[worker_id].recieve_specify(remain_to_receive));
                remain_to_receive=0;
            }//else
        }//while
        mtx.lock();
        aggregate_histogram_perworker(worker_id,batch_id,node_num_startID_curl);
        finish_batch_num[worker_id]+=1;
        mtx.unlock();
    }//for


    if(node_num_startID_curl==int(pow(2.0,LAYER_NUMBER-1-1))){
        return;
    }//if


    string s,rcv_s;

    if(need_feat_num_ahead[worker_id]<pipeline_feature_batch){
        sock[worker_id].send_uint32(0);
        return;
    }//if
    int need_batch_num=need_feat_num_ahead[worker_id]/pipeline_feature_batch;
    if(need_batch_num>=(FEATURE_NUMBER/pipeline_feature_batch))
        need_batch_num=(FEATURE_NUMBER/pipeline_feature_batch)-1;

    
    int cur_batch_received=batch_receive_signal;
    int *return_topN_feature;
    int top_feature_num;
    if(cur_batch_received>=need_batch_num){
        sock[worker_id].send_uint32(1); 
        top_feature_num=need_batch_num*pipeline_feature_batch*step2_feat_frac;
        return_topN_feature=new int[top_feature_num*node_num_startID_curl];
        find_topN_candidate(1,cur_batch_received,top_feature_num,return_topN_feature,node_num_startID_curl);
        sock[worker_id].send_uint32(4*node_num_startID_curl*top_feature_num);

        s="";
        for(int i=0;i<node_num_startID_curl*top_feature_num;i++){
            s+=u2b(return_topN_feature[i]);
        }//for
        sock[worker_id].send_specify(s,4*node_num_startID_curl*top_feature_num);
        delete[] return_topN_feature;
        return;
    }//if

    int processed_batch=cur_batch_received;
    int need_commu_times;

    if(processed_batch!=0)  need_commu_times=1+(need_batch_num-processed_batch);
    else    need_commu_times=need_batch_num-processed_batch;
    sock[worker_id].send_uint32(need_commu_times);


    if(processed_batch!=0){
        top_feature_num=processed_batch*pipeline_feature_batch*step2_feat_frac;
        return_topN_feature=new int[top_feature_num*node_num_startID_curl];
        find_topN_candidate(1,processed_batch,top_feature_num,return_topN_feature,node_num_startID_curl);
        sock[worker_id].send_uint32(4*node_num_startID_curl*top_feature_num);
        s="";
        for(int i=0;i<node_num_startID_curl*top_feature_num;i++){
            s+=u2b(return_topN_feature[i]);
        }//for
        sock[worker_id].send_specify(s,4*node_num_startID_curl*top_feature_num);
        delete[] return_topN_feature;
    }//if
    while(processed_batch<need_batch_num){
        while(1)
            if(batch_receive_signal>processed_batch){
                processed_batch++;
                break;
            }//if

        top_feature_num=pipeline_feature_batch*step2_feat_frac;
        return_topN_feature=new int[top_feature_num*node_num_startID_curl];
        find_topN_candidate(processed_batch,processed_batch,top_feature_num,return_topN_feature,node_num_startID_curl);

        sock[worker_id].send_uint32(4*node_num_startID_curl*top_feature_num);

        s="";
        for(int i=0;i<node_num_startID_curl*top_feature_num;i++){
            s+=u2b(return_topN_feature[i]);
        }//for

        sock[worker_id].send_specify(s,4*node_num_startID_curl*top_feature_num);
        delete[] return_topN_feature;
    }//while
    return;
}//void over



struct Master
{
    uint32_t query_time = 0, query_cnt = 0;
    uint32_t instance_number, eps;
    
    
    double get_l(const vector<pair<uint32_t, double> > &tmp, uint32_t rank)
    {
        if (tmp.back().first < rank) return tmp.back().second;
        int l = 0, r = tmp.size() - 1;
        while (l + 1 < r)
        {
            int mid = (l + r) / 2;
            if (tmp[mid].first < rank) l = mid;
            else r = mid;
        }
        return tmp[l].second;
    }
    double get_r(const vector<pair<uint32_t, double> > &tmp, uint32_t rank)
    {
        if (tmp.front().first > rank) return tmp.front().first;
        int l = 0, r = tmp.size() - 1;
        while (l + 1 < r)
        {
            int mid = (l + r) / 2;
            if (tmp[mid].first > rank) r = mid;
            else l = mid;
        }
        return tmp[r].second;
    }
    
   

    void broadcast_quantiles()
    {
        for (uint i = 0; i < WORKER_NUMBER; i++)
            sock[i].send_uint32(QUANTILE_NUMBER);
        int num = 50 / QUANTILE_NUMBER;
        int t = (FEATURE_NUMBER + num - 1 / num);
        int cur = 0;
        for (uint i = 0; i < t; i++)
        {
            string s;
            for (uint32_t j = 0; cur < FEATURE_NUMBER && j < num; j++)
            {

                for (uint32_t k = 0; k < QUANTILE_NUMBER-1; k++)
                {
                    s += d2b(quantiles[cur][k]);
                }
                cur++;
            }
            for (uint32_t j = 0; j < WORKER_NUMBER; j++)
                sock[j].send_str(s);
        }
    }

    void load_quantiles(){
        quantiles.resize(FEATURE_NUMBER);
        ifstream in_q("quantiles.txt", ios::in);
        assert(in_q.is_open());
        for (uint32_t i = 0; i < FEATURE_NUMBER; i++)
        {
            vector<string> vct1 = read_line(in_q);
            for (uint32_t j = 0; j < vct1.size(); j++)
                quantiles[i].push_back(stod(vct1[j]));
            assert(quantiles[i].size()==QUANTILE_NUMBER-1);
        }//for
    }

    void sync_loading_dataset_finish(){
        double totalnum=0;
        for (uint32_t j = 0; j < WORKER_NUMBER; j++){
            totalnum+=(sock[j].recieve_double());
        }//for
        for (uint32_t j = 0; j < WORKER_NUMBER; j++){
            sock[j].send_double(totalnum);
        }//for
    }


    void train_pre_election()
    {
        double d_sum,d_tmp;
        Binary_Double tmp_g;
        Binary_Double tmp_h;
        vector<pair<double, double>> candidate_split(FEATURE_NUMBER);
        struct timeval myt1,myt2,myt3,myt4,oh_t1,oh_t2;
        double train_time;
        double oh_time;
        int rcv_hist_done_time[LAYER_NUMBER][3];
        double compute_split_time[LAYER_NUMBER];
        thread thread_hist[40];
        vector<double> feature_gain_rank;
        vector<vector<int>> list_of_all_topN_feature;
        vector<int> pre_ele_split_record;
        ofstream of_topN_cdf,of_top1_drift,of_top1_true_feat,of_insurance_feat;
        vector<pair<int,int>> insurance_feature;


        of_topN_cdf.open(middle_result_path+"t10l8_topN_cdf.txt",ios::out);
        of_top1_drift.open(middle_result_path+"t10l8_top1_drift.txt",ios::out);
        of_top1_true_feat.open(middle_result_path+"t10l8_top1_true_ID.txt",ios::out);
        of_insurance_feat.open(middle_result_path+"t10l8_insurance_feat.txt",ios::out);
        assert(of_topN_cdf.is_open());
        

        int redo_layer;
        string true_split_info_for_false_layer;
        int is_redoing=0;



        init_scheduler();

        for (uint32_t i_tree = 1; i_tree <= TREE_NUMBER; i_tree++)
        {   
        
            update_schedule_per_tree_adaptive_fast(i_tree);

            pre_ele_split_record.clear();
            pre_ele_whole_history.resize(int(pow(2.0,LAYER_NUMBER-1)));
            for(int node_id=1;node_id<int(pow(2.0,LAYER_NUMBER-2));node_id++)
                pre_ele_whole_history[node_id].clear();
            
            uint32_t i_layer;

            int remain_to_receive,top_feature_num,node_num_startID_curl;
            int *return_topN_feature;
            string s,schedule_info;
            tm *_tm;
            time_t t;
            double *db_rcv_array[WORKER_NUMBER];


            for(i_layer = 1; i_layer<=LAYER_NUMBER-1; i_layer++){

                if(i_layer==LAYER_NUMBER-1){
                    for(int i_worker=0;i_worker<WORKER_NUMBER;i_worker++){
                        worker_step1_frac[i_worker]=1.0;
                    }//for
                }//if


                for(int i_worker=0;i_worker<WORKER_NUMBER;i_worker++){
                    schedule_info="";
                    schedule_info+=d2b(worker_step1_frac[i_worker]);
                    schedule_info+=d2b(step2_feat_frac);
                    sock[i_worker].send_specify(schedule_info,16);
                }//for

                node_num_startID_curl=int(pow(2.0,i_layer-1));

                agg_histogram_g.resize(node_num_startID_curl,vector<vector<Binary_Double>>(FEATURE_NUMBER,vector<Binary_Double>(QUANTILE_NUMBER)));
                agg_histogram_h.resize(node_num_startID_curl,vector<vector<Binary_Double>>(FEATURE_NUMBER,vector<Binary_Double>(QUANTILE_NUMBER)));
                for(int x=0;x<node_num_startID_curl;x++){
                    for(int y=0;y<FEATURE_NUMBER;y++){
                        for(int z=0;z<QUANTILE_NUMBER;z++){
                            agg_histogram_g[x][y][z].value=0;
                            agg_histogram_h[x][y][z].value=0;
                        }//for
                    }//for
                }//for
                receive_buffer.resize(WORKER_NUMBER);

                for(int w_id=0;w_id<WORKER_NUMBER;w_id++){
                    string().swap(receive_buffer[w_id]);
                }//for

                batch_receive_signal=0;
                finish_batch_num.resize(WORKER_NUMBER);
                for(int w=0;w<WORKER_NUMBER;w++){
                    finish_batch_num[w]=0;
                    thread_hist[w]=thread(thread_receive_hist_pre_ele,node_num_startID_curl,w);
                }//for

                for(int commu_batch_id=0;commu_batch_id<(FEATURE_NUMBER-1)/pipeline_feature_batch+1;commu_batch_id++){
                    for(int w_id=0;w_id<WORKER_NUMBER;w_id++){
                        while(finish_batch_num[w_id]<=commu_batch_id){
                            continue;
                        }//while
                    }//for
                    batch_receive_signal=commu_batch_id+1;
                }//for 
                for(int w=0;w<WORKER_NUMBER;w++)
                    thread_hist[w].join();
                t = time(NULL);
                _tm=localtime(&t);
                gettimeofday(&myt2, NULL);


                if(i_layer==LAYER_NUMBER-1)
                    break;
                gettimeofday(&oh_t1, NULL);

                top_feature_num=FEATURE_NUMBER*step2_feat_frac;
                return_topN_feature=new int[node_num_startID_curl*top_feature_num];
                find_topN_candidate(1,(FEATURE_NUMBER-1)/pipeline_feature_batch+1,top_feature_num,return_topN_feature,node_num_startID_curl,1);
                s="";
                for(int i=0;i<node_num_startID_curl*top_feature_num;i++){
                    s+=u2b(return_topN_feature[i]); 
                }//for
                gettimeofday(&oh_t2, NULL);
                oh_time = (oh_t2.tv_sec - oh_t1.tv_sec) + (double)(oh_t2.tv_usec - oh_t1.tv_usec)/1000000.0;

    
                gettimeofday(&myt3, NULL);
                train_time = (myt3.tv_sec - myt2.tv_sec) + (double)(myt3.tv_usec - myt2.tv_usec)/1000000.0;

                for(int i_worker=0;i_worker<WORKER_NUMBER;i_worker++){
                    sock[i_worker].send_specify(s,4*node_num_startID_curl*top_feature_num);
                }//for
                

                remain_to_receive=2*8*node_num_startID_curl*top_feature_num*QUANTILE_NUMBER;
                for(int i_worker=0;i_worker<WORKER_NUMBER;i_worker++)
                    string().swap(receive_buffer[i_worker]);
                while(remain_to_receive>0){
                    if(remain_to_receive>4194304){
                        for(int i_worker=0;i_worker<WORKER_NUMBER;i_worker++)
                            receive_buffer[i_worker]+=(sock[i_worker].recieve_specify(4194304));
                        remain_to_receive-=4194304;
                    }//if
                    else{
                        for(int i_worker=0;i_worker<WORKER_NUMBER;i_worker++)
                            receive_buffer[i_worker]+=(sock[i_worker].recieve_specify(remain_to_receive));
                        remain_to_receive=0;
                    }//else
                }//while

                for(int i_worker=0;i_worker<WORKER_NUMBER;i_worker++)
                    db_rcv_array[i_worker]=(double*)(receive_buffer[i_worker].c_str());

                
                for(uint32_t i_node = 0; i_node < node_num_startID_curl; i_node++){
                    for (int j_count=0;j_count<top_feature_num;j_count++){
                        int j=return_topN_feature[i_node*top_feature_num+j_count];
                        for (uint32_t q = 0; q < QUANTILE_NUMBER; q++){
                            for(int i_worker=0;i_worker<WORKER_NUMBER;i_worker++){
                                tmp_g.value=*(db_rcv_array[i_worker]+2*(i_node*top_feature_num*QUANTILE_NUMBER+j_count*QUANTILE_NUMBER+q));
                                tmp_h.value=*(db_rcv_array[i_worker]+2*(i_node*top_feature_num*QUANTILE_NUMBER+j_count*QUANTILE_NUMBER+q)+1);
                                agg_histogram_g[i_node][j][q].value+=tmp_g.value;
                                agg_histogram_h[i_node][j][q].value+=tmp_h.value;
                                
                            }//for
                        }//for
                    }//for
                }//for

                s="";
                for(uint32_t i_node = 0; i_node < node_num_startID_curl; i_node++){

                    for (int j_count=0;j_count<top_feature_num;j_count++){
                        int j=return_topN_feature[i_node*top_feature_num+j_count];
                        uint32_t cur;
                        split_finding(agg_histogram_g[i_node][j], agg_histogram_h[i_node][j], my_lambda, cur, candidate_split[j].second);
                        candidate_split[j].first = cur;
                    }//for 
                    uint32_t pos = return_topN_feature[0+i_node*top_feature_num];
                    for (int j_count=0;j_count<top_feature_num;j_count++){
                        int j=return_topN_feature[i_node*top_feature_num+j_count];
                        if (candidate_split[j].second>(candidate_split[pos].second+my_EPS))
                            pos = j;
                    }//for
                    s += u2b(pos);
                    s += d2b(candidate_split[pos].first);
                    s += d2b(candidate_split[pos].second);
                    pre_ele_split_record.push_back(pos);

                }//for 

                if(0){
                    redo_point: s=true_split_info_for_false_layer;
                    i_layer=redo_layer;
                    node_num_startID_curl=int(pow(2.0,i_layer-1));
                    reset_scheduler_for_redo();
                    is_redoing=1;
                    cout<<endl<<"begin redoing now! send split point for layer "<<redo_layer<<endl<<endl;;
                }//if


                for(uint32_t i_node = 0; i_node < node_num_startID_curl; i_node++){
                    for (uint32_t w = 0; w < WORKER_NUMBER; w++)
                        sock[w].send_str(s.substr(i_node*20,20));
                }//for
            }//for 

            for(i_layer = LAYER_NUMBER-1; i_layer<LAYER_NUMBER; i_layer++){
                node_num_startID_curl=int(pow(2.0,i_layer-1));
                gettimeofday(&myt2, NULL);
                s="";
                for(uint32_t i_node = 0; i_node < node_num_startID_curl; i_node++){
                    for (uint32_t j = 0; j < FEATURE_NUMBER; j++){
                        uint32_t cur;
                        split_finding(agg_histogram_g[i_node][j], agg_histogram_h[i_node][j], my_lambda, cur, candidate_split[j].second);
                        candidate_split[j].first = cur;
                    }
                    uint32_t pos = 0;
                    for (uint32_t j = 0; j < FEATURE_NUMBER; j++)
                        if (candidate_split[pos].second < candidate_split[j].second)
                            pos = j;
                    s += u2b(pos);
                    s += d2b(candidate_split[pos].first);
                    s += d2b(candidate_split[pos].second);
                }//for
                gettimeofday(&myt3, NULL);
                train_time = (myt3.tv_sec - myt2.tv_sec) + (double)(myt3.tv_usec - myt2.tv_usec)/1000000.0;
                compute_split_time[i_layer]=train_time;
                for(uint32_t i_node = 0; i_node < node_num_startID_curl; i_node++){
                    for (uint32_t w = 0; w < WORKER_NUMBER; w++)
                        sock[w].send_str(s.substr(i_node*20,20));
                }//for

                if(is_redoing){
                    is_redoing=0;
                    for(int w_id=0;w_id<WORKER_NUMBER;w_id++){
                        s="0";
                        sock[w_id].send_str(s);
                    }//for
                    break;
                }//if

                gettimeofday(&oh_t1, NULL);
                vector<int> top1_true_feat_drift_from_pre_ele;
                vector<int> top1_true_featID_per_tree;
                top1_true_feat_drift_from_pre_ele.clear();
                top1_true_featID_per_tree.clear();
                redo_layer=-1;
                for(uint32_t verify_layer = LAYER_NUMBER-2; verify_layer >=1; verify_layer--){
                    for(int verify_node=0;verify_node<int(pow(2.0,verify_layer-1));verify_node++){
                        for (uint32_t j = 0; j < FEATURE_NUMBER; j++){
                            for (uint32_t q = 0; q < QUANTILE_NUMBER; q++){
                                agg_histogram_g[verify_node][j][q].value=agg_histogram_g[verify_node*2][j][q].value+agg_histogram_g[verify_node*2+1][j][q].value;
                                agg_histogram_h[verify_node][j][q].value=agg_histogram_h[verify_node*2][j][q].value+agg_histogram_h[verify_node*2+1][j][q].value;
                            }//for
                        }//for
                    }//for
                    string temp_split_info="";

                    vector<pair<int, double>> per_node_true_ranks(FEATURE_NUMBER);
                    vector<int> per_node_true_ranks_int(FEATURE_NUMBER);
                    for(int verify_node=0;verify_node<int(pow(2.0,verify_layer-1));verify_node++){
                        int global_node_idx=verify_node+int(pow(2.0,verify_layer-1));
                        per_node_true_ranks.clear();
                        for (uint32_t j = 0; j < FEATURE_NUMBER; j++){
                            uint32_t cur;
                            split_finding(agg_histogram_g[verify_node][j], agg_histogram_h[verify_node][j], my_lambda, cur, candidate_split[j].second);
                            candidate_split[j].first = cur;
                            per_node_true_ranks.push_back(make_pair(j,candidate_split[j].second));
                        }//for
                        uint32_t pos = 0;
                        for (uint32_t j = 0; j < FEATURE_NUMBER; j++)
                            if (candidate_split[j].second>(candidate_split[pos].second+my_EPS))
                                pos = j;

                        temp_split_info+=u2b(pos);
                        temp_split_info+=d2b(candidate_split[pos].first);
                        temp_split_info+=d2b(candidate_split[pos].second);

                        top1_true_featID_per_tree.push_back(pos);
                        int exist_flag=0;
                        for(int ins_idx=0;ins_idx<insurance_feature.size();ins_idx++){
                            if(insurance_feature[ins_idx].first==pos){
                                (insurance_feature[ins_idx].second)++;
                                exist_flag=1;
                                break;
                            }//if
                        }//for
                        if(exist_flag!=1)
                            insurance_feature.push_back(make_pair(pos,1));
                        
                        if(pre_ele_split_record[verify_node+int(pow(2.0,verify_layer-1))-1]!=pos){
                            redo_layer=verify_layer;
                        }//if
                        int top1_feat_rank_in_pre_ele=
                            find(pre_ele_whole_history[global_node_idx].begin(),pre_ele_whole_history[global_node_idx].end(),pos)-pre_ele_whole_history[global_node_idx].begin();
                        top1_true_feat_drift_from_pre_ele.push_back(top1_feat_rank_in_pre_ele+1);
                        sort(per_node_true_ranks.begin(),per_node_true_ranks.end(),cmp);
                        per_node_true_ranks_int.clear();
                        for(int j=0;j<FEATURE_NUMBER;j++) 
                            per_node_true_ranks_int.push_back(per_node_true_ranks[j].first);
                        

                        for(int rank_j=0;rank_j<100;rank_j++){
                            int rank_j_feat_id=per_node_true_ranks_int[rank_j];
                            int predict_rank=find(pre_ele_whole_history[global_node_idx].begin(),pre_ele_whole_history[global_node_idx].end(),rank_j_feat_id)
                                                - pre_ele_whole_history[global_node_idx].begin();
                            of_topN_cdf<<abs(predict_rank-rank_j)<<",";
                        }//for
                        of_topN_cdf<<endl;
                    }//for

                    
                    if(redo_layer==verify_layer)
                        true_split_info_for_false_layer=temp_split_info;

                }//for
                sort(insurance_feature.begin(),insurance_feature.end(),cmp);
                for(int i=0;i<insurance_feature.size();i++){
                    of_insurance_feat<<insurance_feature[i].first<<",";
                }//for
                of_insurance_feat<<endl;


                for(int i=0;i<top1_true_feat_drift_from_pre_ele.size();i++){
                    cout<<top1_true_feat_drift_from_pre_ele[i]<<",";
                    of_top1_drift<<top1_true_feat_drift_from_pre_ele[i]<<",";
                }//for
                of_top1_drift<<endl;

                
                for(int i=0;i<top1_true_featID_per_tree.size();i++){
                    of_top1_true_feat<<top1_true_featID_per_tree[i]<<",";
                }//for
                of_top1_true_feat<<endl;
                cout<<endl;
                gettimeofday(&myt4, NULL);
                train_time = (myt4.tv_sec - myt3.tv_sec) + (double)(myt4.tv_usec - myt3.tv_usec)/1000000.0;
                gettimeofday(&oh_t2, NULL);
                oh_time = (oh_t2.tv_sec - oh_t1.tv_sec) + (double)(oh_t2.tv_usec - oh_t1.tv_usec)/1000000.0;
                
                s="";
                
                if(redo_layer!=-1){
                    for(int w_id=0;w_id<WORKER_NUMBER;w_id++){
                        s="1";
                        sock[w_id].send_str(s);
                        s=u2b(redo_layer);
                        sock[w_id].send_str(s);
                    }//for
                    goto redo_point;
                }//if

                s="0";
                for(int w_id=0;w_id<WORKER_NUMBER;w_id++){
                    sock[w_id].send_str(s);
                }//for

            }//for 

            double allsum_g,allsum_h;
            for (uint32_t i_node = int(pow(2,LAYER_NUMBER-1)); i_node < int(pow(2,LAYER_NUMBER)); i_node++){
                allsum_g=0;
                allsum_h=0;
                for (uint32_t j = 0; j < WORKER_NUMBER; j++){
                    allsum_g+=(sock[j].recieve_double());
                    allsum_h+=(sock[j].recieve_double());
                }//for
                for (uint32_t j = 0; j < WORKER_NUMBER; j++){
                    sock[j].send_double(allsum_g);
                    sock[j].send_double(allsum_h);
                }//for
            }//for
        }


        of_topN_cdf.close();
        of_top1_drift.close();
        of_top1_true_feat.close();
        of_insurance_feat.close();
       
    }
    
};




int main(int argc, char** argv)
{
    assert(argc == 7);
    WORKER_NUMBER = stoul(argv[1]);
    QUANTILE_NUMBER = stod(argv[2]);
    FEATURE_NUMBER = stoul(argv[3]);
    TREE_NUMBER=stoul(argv[4]);
    LAYER_NUMBER=stoul(argv[5]);
    server_in_bandwidth = stod(argv[6]);

    struct timeval t1,t2;
    gettimeofday(&t1, NULL);
    char self_addr[]="172.23.0.1";
    char w_addr[8][14]={"172.23.0.1","172.23.12.100","172.23.12.105","172.23.12.110","172.23.12.108","172.23.12.141","172.23.12.142","172.23.12.127"};
    int server_port[8]={40000,40001,40002,40003,40004,40005,40006,40007};
    

    ifstream avoid_overwrite_in1(middle_result_path + "t10l8_topN_cdf.txt", ios::in);
    ifstream avoid_overwrite_in2(middle_result_path + "t10l8_top1_drift.txt", ios::in);
    assert(!(avoid_overwrite_in1.is_open()));
    assert(!(avoid_overwrite_in2.is_open()));
    avoid_overwrite_in1.close();
    avoid_overwrite_in2.close();

    sock[0].init();
    sock[0].link(w_addr[0],self_addr,30000,40000);
    sock[1].init();
    sock[1].link(w_addr[1],self_addr,30000,40001);
    sock[2].init();
    sock[2].link(w_addr[2],self_addr,30000,40002);
    sock[3].init();
    sock[3].link(w_addr[3],self_addr,30000,40003);
    sock[4].init();
    sock[4].link(w_addr[4],self_addr,30000,40004);
    sock[5].init();
    sock[5].link(w_addr[5],self_addr,30000,40005);
    sock[6].init();
    sock[6].link(w_addr[6],self_addr,30000,40006);
    sock[7].init();
    sock[7].link(w_addr[7],self_addr,30000,40007);


    

    Master master;

    master.sync_loading_dataset_finish();

    master.load_quantiles();
    


    master.train_pre_election();
    
    
    for (uint32_t i = 0; i < WORKER_NUMBER; i++)
        sock[i].close();


    return 0;
}
