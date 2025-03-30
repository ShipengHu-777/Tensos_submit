#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include "typedefs.h"
#include "socket.h"
#include "connection.h"
#include <cmath>
#include "time.h"
#include <thread>
#include <array>

using namespace std;
const uint32_t FIELD_SIZE = 1024;
uint32_t INSTANCE_NUMBER;
uint32_t test_INSTANCE_NUMBER;
uint32_t FEATURE_NUMBER;
uint32_t QUANTILE_NUMBER;
uint32_t TREE_NUMBER;
uint32_t LAYER_NUMBER;
uint32_t PRE_ELECTION_OPTIM = 1;
int read_file_batch=4000;

int MAX_THREAD_NUMBER;
uint32_t WORKER_NUMBER;
const char *address = "127.0.0.1";
const double eps = 1e-6;
const double my_lambda=0.05;
const double learning_rate=0.5;
const uint32_t MAX_WORKER_NUMBER = 32;
int pipeline_feature_batch=150;
int least_compute_feat_num=80;
int exp_all_correct=0;
const uint32_t BLOCK = 10;
uint16_t port_in, port_out;
string file;
string key_file;
string test_file;

uint32_t id;

double first_order_gradient(double real_value, double predict_value)
{
    return predict_value-real_value;
    
}

double second_order_gradient(double real_value, double predict_value)
{
    return predict_value*(1-predict_value);
    
}

void split_finding(vector<double> G, vector<double> H, double lambda, uint32_t &split_point, double &max_score)
{
    assert(G.size() == H.size());
    double Gl = 0, Hl = 0;
    double Gsum = 0;
    for (const auto &p : G) Gsum += p;
    double Hsum = 0;
    for (const auto &p : H) Hsum += p;
    uint32_t candidate_split_point = 0;
    double current_max_score = 0.0;
    for (uint32_t i = 0; i < G.size(); i++)
    {
        Gl += G[i];
        Hl += H[i];
        double Gr = Gsum - Gl;
        double Hr = Hsum - Hl;
        double tmp_score = Gl * Gl / (Hl + lambda) + Gr * Gr / (Hr + lambda) - Gsum * Gsum / (Hsum + lambda);
        if (tmp_score > current_max_score)
        {
            candidate_split_point = i;
            current_max_score = tmp_score;
        }
    }
    split_point = candidate_split_point;
    max_score = current_max_score;
    return;
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

union Binary_Float
{
    float value;
    char s[4];
}binary_float;

union Binary_Double
{
    double value;
    char s[8];
}binary_double;

union Binary_Uint32
{
    uint32_t value;
    char s[4];
}binary_uint;

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

struct Sock
{
    CSocket sock_in, sock_out;
    uint32_t send_cnt = 0, recv_cnt = 0;
    void link(char* server_addr,char* self_addr)
    {
        listen(self_addr, port_in, &sock_in, 1);
        connect(server_addr, port_out, sock_out);
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
        send_cnt++;
        sock_out.Send(s, FIELD_SIZE);
    }
    void send_str(string s)
    {
        send_cstr(s.c_str());
    }

    void send_double(double x)
    {
        send_str(d2b(x));
    }
    void send_uint32(uint32_t x)
    {
        send_str(u2b(x));
    }
    
    void recieve_cstr(char *s)
    {
        recv_cnt++;
        sock_in.Receive(s, FIELD_SIZE);
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

    string recieve_specify(int my_len){
        char s[my_len+1];
        sock_in.Receive(s, my_len+1);
        string str;
        for (uint32_t i = 0; i < my_len; i++)
            str += s[i];
        return str;
    }
    
    uint32_t recieve_uint()
    {
        return b2u(recieve_str(4));
    }

    double recieve_double()
    {
        return b2d(recieve_str(8));
    }

   
}sock;


struct Instance
{
    string name;
    double label;
    vector<double> value;
    Instance () {}
};




vector<vector<double>> quantiles;
vector<vector<vector<Binary_Double>>> bucket_g;
vector<vector<vector<Binary_Double>>> bucket_h;
vector<Instance> instance;
vector<vector<double>> temp_values;
vector<int> inst_labels;
vector<vector<uint16_t>> instances_bucket;
vector<int> inst_gh_mapping;
vector<vector<vector<int>>> bucket_gh_meta;
vector<vector<int>> node;
vector<double> g_inst, h_inst;
vector<double> unique_g, unique_h;
vector<int> split_feature;
vector<int> split_bkt;
vector<vector<int>> split_result;
vector<string> send_s;
vector<vector<int>> list_of_all_topN_feature;
vector<vector<vector<Binary_Double>>> thread_bucket_g;
vector<vector<vector<Binary_Double>>> thread_bucket_h;
vector<vector<vector<Binary_Double>>> thread_continue_bucket_g;
vector<vector<vector<Binary_Double>>> thread_continue_bucket_h;
vector<int> hist_comp_thread_status;
int to_comp_feature_num[16];
vector<vector<double>> mini_bucket_g;
vector<vector<double>> mini_bucket_h;



uint32_t get_bucket_id(double x, uint32_t id)
{
    return lower_bound(quantiles[id].begin(), quantiles[id].end(), x) - quantiles[id].begin();
}


void thread_node_split(int split_node_id, int node_num_startID_curl, int thread_id, int total_thread_num){
    int total_num=node[split_node_id].size();
    int i;
    for(i=thread_id*(total_num/total_thread_num); i<(thread_id+1)*(total_num/total_thread_num);i++){
        if (instances_bucket[node[split_node_id][i]][split_feature[split_node_id-node_num_startID_curl]] <= split_bkt[split_node_id-node_num_startID_curl]) 
            split_result[thread_id].push_back(0);
        else 
            split_result[thread_id].push_back(1);
    }//for
    if(thread_id==total_thread_num-1){
        while(i<total_num){
            if (instances_bucket[node[split_node_id][i]][split_feature[split_node_id-node_num_startID_curl]] <= split_bkt[split_node_id-node_num_startID_curl]) 
                split_result[thread_id].push_back(0);
            else 
                split_result[thread_id].push_back(1);
            i++;
        }//while
    }//if
}


void thread_hist_const_new(int node_num_startID_curl,
    int thread_id,int total_thread_num){
        int total_inst_num=0;
        uint32_t jstart, jend;
        jstart=thread_id*FEATURE_NUMBER/total_thread_num;
        jend=(thread_id+1)*FEATURE_NUMBER/total_thread_num;
        for (uint32_t i_node = 0; i_node < node_num_startID_curl; i_node++){
            for (auto &p : node[i_node+node_num_startID_curl]){
                for (uint32_t j = jstart; j < jend; j++)
                {
                    if(j>=FEATURE_NUMBER)
                        break;
                    bucket_g[i_node][j][instances_bucket[p][j]].value += g_inst[p];
                    bucket_h[i_node][j][instances_bucket[p][j]].value += h_inst[p];
                }//for
            }//for
        }//for 
}


void thread_hist_const_new_pre_ele_first_sparse(int node_num_startID_curl,
    int thread_id,int total_thread_num, double whole_comp_frac){
        
        vector<vector<Binary_Double>> zero_opt_node_sum;
        zero_opt_node_sum.resize(node_num_startID_curl,vector<Binary_Double>(2));
        for(int i=0;i<node_num_startID_curl;i++){
            zero_opt_node_sum[i][0].value=0;
            zero_opt_node_sum[i][1].value=0;
        }//for

        int to_comp_inst_num_per_node=0;
        uint32_t j_start, j_end;
        int cur_node_count;        
        int per_thread_inst=(INSTANCE_NUMBER*whole_comp_frac)/total_thread_num;
        int condition_1=per_thread_inst*thread_id;
        int thread_inst_count=0;
        int total_inst_for_step1=0;
        
        for(int batch_id=0;batch_id<(FEATURE_NUMBER-1)/pipeline_feature_batch+1;batch_id++){
            j_start=batch_id*pipeline_feature_batch;
            j_end=(batch_id+1)*pipeline_feature_batch;
            if(j_end>FEATURE_NUMBER)   j_end=FEATURE_NUMBER;

            thread_inst_count=0;
            total_inst_for_step1=0;
            for (uint32_t i_node = 0; i_node < node_num_startID_curl; i_node++){
                if(thread_inst_count>=per_thread_inst && thread_id!=(total_thread_num-1))
                    break;
                int thread_inode_idx=thread_id*node_num_startID_curl+i_node;
                to_comp_inst_num_per_node=(node[i_node+node_num_startID_curl].size())*whole_comp_frac;

                cur_node_count=0;
                for (auto &p : node[i_node+node_num_startID_curl]){
                    cur_node_count++;
                    if(cur_node_count>to_comp_inst_num_per_node)
                        break;
                    if(thread_inst_count>=per_thread_inst && thread_id!=(total_thread_num-1))
                        break;
                    if(total_inst_for_step1>=condition_1){
                        if(batch_id==0){
                            zero_opt_node_sum[i_node][0].value+=g_inst[p];
                            zero_opt_node_sum[i_node][1].value+=h_inst[p];
                        }//if
                        for (uint32_t j = j_start; j < j_end; j++)
                        {
                            if(instances_bucket[p][j]==0)   continue;
                            thread_bucket_g[thread_inode_idx][j][instances_bucket[p][j]].value += g_inst[p];
                            thread_bucket_h[thread_inode_idx][j][instances_bucket[p][j]].value += h_inst[p];
                        }//for
                        thread_inst_count++;
                    }//for
                    total_inst_for_step1++;
                }//for
                for (uint32_t j = j_start; j < j_end; j++)
                {
                    thread_bucket_g[thread_inode_idx][j][0].value += zero_opt_node_sum[i_node][0].value;
                    thread_bucket_h[thread_inode_idx][j][0].value += zero_opt_node_sum[i_node][1].value;
                    for(int q=1;q<QUANTILE_NUMBER;q++){
                        thread_bucket_g[thread_inode_idx][j][0].value -= thread_bucket_g[thread_inode_idx][j][q].value;
                        thread_bucket_h[thread_inode_idx][j][0].value -= thread_bucket_h[thread_inode_idx][j][q].value;
                    }//for
                }//for
            }//for 
            hist_comp_thread_status[thread_id]=batch_id+1;
        }//for
        
}


void thread_hist_const_new_pre_ele_first(int node_num_startID_curl,
    int thread_id,int total_thread_num, double whole_comp_frac){
        
        int to_comp_inst_num_per_node=0;
        uint32_t j_start, j_end;
        int cur_node_count;        
        int per_thread_inst=(INSTANCE_NUMBER*whole_comp_frac)/total_thread_num;
        int condition_1=per_thread_inst*thread_id;
        int thread_inst_count=0;
        int total_inst_for_step1=0;
        
        for(int batch_id=0;batch_id<(FEATURE_NUMBER-1)/pipeline_feature_batch+1;batch_id++){
            j_start=batch_id*pipeline_feature_batch;
            j_end=(batch_id+1)*pipeline_feature_batch;
            if(j_end>FEATURE_NUMBER)   j_end=FEATURE_NUMBER;

            thread_inst_count=0;
            total_inst_for_step1=0;
            for (uint32_t i_node = 0; i_node < node_num_startID_curl; i_node++){
                if(thread_inst_count>=per_thread_inst && thread_id!=(total_thread_num-1))
                    break;
                int thread_inode_idx=thread_id*node_num_startID_curl+i_node;

                to_comp_inst_num_per_node=(node[i_node+node_num_startID_curl].size())*whole_comp_frac;

                cur_node_count=0;
                for (auto &p : node[i_node+node_num_startID_curl]){
                    cur_node_count++;
                    if(cur_node_count>to_comp_inst_num_per_node)
                        break;
                    if(thread_inst_count>=per_thread_inst && thread_id!=(total_thread_num-1))
                        break;
                    if(total_inst_for_step1>=condition_1){
                        for (uint32_t j = j_start; j < j_end; j++)
                        {
                            thread_bucket_g[thread_inode_idx][j][instances_bucket[p][j]].value += g_inst[p];
                            thread_bucket_h[thread_inode_idx][j][instances_bucket[p][j]].value += h_inst[p];
                        }//for
                        thread_inst_count++;
                    }//for
                    total_inst_for_step1++;
                }//for
            }//for 
            hist_comp_thread_status[thread_id]=batch_id+1;
        }//for
        
}//void


void thread_hist_const_new_pre_ele_second_sparse(int node_num_startID_curl,
    int thread_id,int total_thread_num, double whole_comp_frac){   
        vector<vector<Binary_Double>> zero_opt_node_sum;
        zero_opt_node_sum.resize(node_num_startID_curl,vector<Binary_Double>(2));
        for(int i=0;i<node_num_startID_curl;i++){
            zero_opt_node_sum[i][0].value=0;
            zero_opt_node_sum[i][1].value=0;
        }//for

        assert(total_thread_num>1);
        vector<vector<double>> mini_bucket_g;
        vector<vector<double>> mini_bucket_h;

        mini_bucket_g.resize(FEATURE_NUMBER,vector<double>(QUANTILE_NUMBER));
        mini_bucket_h.resize(FEATURE_NUMBER,vector<double>(QUANTILE_NUMBER));
        for(int jj=0;jj<FEATURE_NUMBER;jj++)
            for(int qq=0;qq<QUANTILE_NUMBER;qq++){
                mini_bucket_g[jj][qq]=0;
                mini_bucket_h[jj][qq]=0;
            }//for

        int skip_comp_inst_num_per_node;
        int cur_count_per_node,total_count,thread_inst_total_count;
        int per_thread_inst=(INSTANCE_NUMBER*(1-whole_comp_frac))/total_thread_num;
        total_count=0;
        thread_inst_total_count=0;
        int condition_1=per_thread_inst*thread_id;

        for (uint32_t i_node = 0; i_node < node_num_startID_curl; i_node++){
            if(thread_inst_total_count>=per_thread_inst && thread_id!=(total_thread_num-1))
                break;
            int thread_inode_idx=thread_id*node_num_startID_curl+i_node;
            to_comp_feature_num[thread_id]=list_of_all_topN_feature[i_node].size();
            int real_j[to_comp_feature_num[thread_id]];
            for (uint32_t j = 0; j < to_comp_feature_num[thread_id]; j++)
                real_j[j]=list_of_all_topN_feature[i_node][j];
            skip_comp_inst_num_per_node=(node[i_node+node_num_startID_curl].size())*whole_comp_frac;
            cur_count_per_node=0;
 
            for (auto &p : node[i_node+node_num_startID_curl]){
                cur_count_per_node++;
                if(cur_count_per_node<=skip_comp_inst_num_per_node)
                    continue;
                if(thread_inst_total_count>=per_thread_inst && thread_id!=(total_thread_num-1))
                    break;
                if(total_count>=condition_1){
                    zero_opt_node_sum[i_node][0].value+=g_inst[p];
                    zero_opt_node_sum[i_node][1].value+=h_inst[p];
                    int j;
                    for(int j_idx=0;j_idx<to_comp_feature_num[thread_id];j_idx++)
                    {
                        j=real_j[j_idx];
                        if(instances_bucket[p][j]==0)   continue;
                        mini_bucket_g[j_idx][instances_bucket[p][j]] += g_inst[p];
                        mini_bucket_h[j_idx][instances_bucket[p][j]] += h_inst[p];
                    }//for
                    thread_inst_total_count++;
                }//if
                total_count++;
            }//for 

            for(int j_idx=0;j_idx<to_comp_feature_num[thread_id];j_idx++){
                int j=real_j[j_idx];
                mini_bucket_g[j_idx][0]+=zero_opt_node_sum[i_node][0].value;
                mini_bucket_h[j_idx][0]+=zero_opt_node_sum[i_node][1].value;
                for(int q=1;q<QUANTILE_NUMBER;q++){
                    mini_bucket_g[j_idx][0]-=mini_bucket_g[j_idx][q];
                    mini_bucket_h[j_idx][0]-=mini_bucket_h[j_idx][q];
                }//for
            }//for

            for(int j_idx=0;j_idx<to_comp_feature_num[thread_id];j_idx++){
                int j=real_j[j_idx];
                for(int q=0;q<QUANTILE_NUMBER;q++){
                    thread_bucket_g[thread_inode_idx][j][q].value=mini_bucket_g[j_idx][q];
                    thread_bucket_h[thread_inode_idx][j][q].value=mini_bucket_h[j_idx][q];
                    mini_bucket_g[j_idx][q]=0;
                    mini_bucket_h[j_idx][q]=0;
                }//for
            }//for

        }//for

        
}


void thread_hist_const_new_pre_ele_second(int node_num_startID_curl,
    int thread_id,int total_thread_num, double whole_comp_frac){   
        
        assert(total_thread_num>1);
        vector<vector<double>> mini_bucket_g;
        vector<vector<double>> mini_bucket_h;


        mini_bucket_g.resize(FEATURE_NUMBER,vector<double>(QUANTILE_NUMBER));
        mini_bucket_h.resize(FEATURE_NUMBER,vector<double>(QUANTILE_NUMBER));
        for(int jj=0;jj<FEATURE_NUMBER;jj++)
            for(int qq=0;qq<QUANTILE_NUMBER;qq++){
                mini_bucket_g[jj][qq]=0;
                mini_bucket_h[jj][qq]=0;
            }//for

        int skip_comp_inst_num_per_node;
        int cur_count_per_node,total_count,thread_inst_total_count;
        int per_thread_inst=(INSTANCE_NUMBER*(1-whole_comp_frac))/total_thread_num;
        total_count=0;
        thread_inst_total_count=0;
        int condition_1=per_thread_inst*thread_id;

        for (uint32_t i_node = 0; i_node < node_num_startID_curl; i_node++){
            if(thread_inst_total_count>=per_thread_inst && thread_id!=(total_thread_num-1))
                break;
            int thread_inode_idx=thread_id*node_num_startID_curl+i_node;
            to_comp_feature_num[thread_id]=list_of_all_topN_feature[i_node].size();
            int real_j[to_comp_feature_num[thread_id]];
            for (uint32_t j = 0; j < to_comp_feature_num[thread_id]; j++)
                real_j[j]=list_of_all_topN_feature[i_node][j];
            skip_comp_inst_num_per_node=(node[i_node+node_num_startID_curl].size())*whole_comp_frac;
            cur_count_per_node=0;
 
            for (auto &p : node[i_node+node_num_startID_curl]){
                cur_count_per_node++;
                if(cur_count_per_node<=skip_comp_inst_num_per_node)
                    continue;
                if(thread_inst_total_count>=per_thread_inst && thread_id!=(total_thread_num-1))
                    break;
                if(total_count>=condition_1){
                    int j;
                    for(int j_idx=0;j_idx<to_comp_feature_num[thread_id];j_idx++)
                    {
                        j=real_j[j_idx];
                        mini_bucket_g[j_idx][instances_bucket[p][j]] += g_inst[p];
                        mini_bucket_h[j_idx][instances_bucket[p][j]] += h_inst[p];
                    }//for
                    thread_inst_total_count++;
                }//if
                total_count++;
            }//for 

            for(int j_idx=0;j_idx<to_comp_feature_num[thread_id];j_idx++){
                int j=real_j[j_idx];
                for(int q=0;q<QUANTILE_NUMBER;q++){
                    thread_bucket_g[thread_inode_idx][j][q].value=mini_bucket_g[j_idx][q];
                    thread_bucket_h[thread_inode_idx][j][q].value=mini_bucket_h[j_idx][q];
                    mini_bucket_g[j_idx][q]=0;
                    mini_bucket_h[j_idx][q]=0;
                }//for
            }//for

        }//for 
      
}



void thread_inst_link_bkt_new_fd_orig(int id_thread,int read_file_batch,int i){
    int start_id;
    if((i+1)%read_file_batch!=0)
        start_id=((i+1)/read_file_batch)*read_file_batch;
    else
        start_id=((i+1)/read_file_batch-1)*read_file_batch;
    for(int tempv_i = id_thread*read_file_batch/32;tempv_i < (id_thread+1)*read_file_batch/32;tempv_i++){
        if((i+1)%read_file_batch!=0 && tempv_i>((i+1)%read_file_batch-1))
            break;
        for(int j=0;j<900;j++){
            int bkt_id=get_bucket_id(temp_values[tempv_i][j], j);
            instances_bucket[tempv_i+start_id].push_back(uint16_t(bkt_id));
        }//for
    }//for
}


struct Worker
{
    
    vector<string> feature_name;
    vector<vector<double>> predict;
    vector<vector<pair<uint32_t, double>>> split_point;
    vector<vector<double>> features;
    Worker ()
    {
    }

    
    void init_fd_orig()
    {
        read_file_batch=8000;
        temp_values.resize(read_file_batch);
        instances_bucket.resize(INSTANCE_NUMBER);
        predict.resize(TREE_NUMBER + 1);
        inst_gh_mapping.resize(INSTANCE_NUMBER);
        for (auto &p : predict) p.resize(INSTANCE_NUMBER);
        split_point.resize(TREE_NUMBER+1);
        for (auto &p : split_point) p.resize(int(pow(2.0,LAYER_NUMBER)));
        ifstream in(file, ios::in);
        assert(in.is_open());
        double tmp_label;
        double train_time;
        struct timeval myt0,myt1;
        string word;
        istringstream sin;
        string line;
        getline(in,line);
        for (uint32_t i = 0; i < INSTANCE_NUMBER; i++)
        {
            getline(in,line);
            sin.clear();
            sin.str(line);
            getline(sin,word,',');
            getline(sin,word,',');
            if(word=="1")
                inst_labels.push_back(1);
            else if(word=="0")
                inst_labels.push_back(0);

            for(int j=0;j<900;j++){
                getline(sin, word, ',');
                temp_values[i%read_file_batch].push_back(stod(word));
            }//for

            if((i+1)%read_file_batch==0 || (i+1)==INSTANCE_NUMBER){
                gettimeofday(&myt1, NULL);
                thread thread_bkt[32];
                for(int id_thread=0;id_thread<32;id_thread++){
                    thread_bkt[id_thread]=thread(thread_inst_link_bkt_new_fd_orig,id_thread,read_file_batch,i);
                }//for
                for(int id_thread=0;id_thread<32;id_thread++){
                    thread_bkt[id_thread].join();
                }//for
                for(int subi=0;subi<read_file_batch;subi++)
                    temp_values[subi].clear();
                gettimeofday(&myt0, NULL);
            }//if
        }//for
    }

   
    uint32_t answer_BS(const uint32_t &id, const double &value)
    {
        return upper_bound(features[id].begin(), features[id].end(), value) - features[id].begin();
    }
    
    void recieve_quantiles()
    {
        quantiles.resize(FEATURE_NUMBER);
        int quantile_number = sock.recieve_uint();
        int num = 50 / QUANTILE_NUMBER;
        int t = (FEATURE_NUMBER + num - 1 / num);
        int cur = 0;
        for (uint i = 0; i < t; i++)
        {
            string s = sock.recieve_str(1000);
            for (uint32_t j = 0; cur < FEATURE_NUMBER && j < num; j++)
            {
                for (uint32_t k = 0; k < quantile_number-1; k++)
                {
                    quantiles[cur].push_back(b2d(s.substr(j*(quantile_number-1)*8 + k*8, 8)));
                }
                cur++;
            }//for
        }
        for (uint32_t i = 0; i < FEATURE_NUMBER; i++)
        {
            for (uint32_t j = 0; j < quantile_number-1; j++)
                cout << quantiles[i][j] << "  ";
            puts("");
        }
    }

    void load_quantiles(){
        quantiles.resize(FEATURE_NUMBER);
        ifstream in_q("persist_quantiles.txt", ios::in);
        for (uint32_t i = 0; i < FEATURE_NUMBER; i++)
        {
            vector<string> vct1 = read_line(in_q);
            for (uint32_t j = 0; j < vct1.size(); j++)
                quantiles[i].push_back(stod(vct1[j]));
        }//for
    }

    void sync_loading_dataset_finish(){
        double totalnum=0;
        sock.send_double(totalnum);
        totalnum=sock.recieve_double();
    }

    void train_pre_election()
    {
        vector<double> all_train_time;
        vector<double> all_layer_time;
        cout << "train" << endl;
        for (uint32_t i = 0; i < INSTANCE_NUMBER; i++)
            predict[0][i] = 0;
        node.resize(int(pow(2.0,LAYER_NUMBER)));
        g_inst.resize(INSTANCE_NUMBER);
        h_inst.resize(INSTANCE_NUMBER);
        vector<pair<double, double>> candidate_split(FEATURE_NUMBER);
        int bucket_tmp_count[QUANTILE_NUMBER];
        struct timeval myt0,myt1,myt2,myt3,myt4,myt5,myt6,myt7,myt8,myt9,myt10,myt00,myt01;
        double train_time;
        double layer_total_training_time[LAYER_NUMBER];
        double hist_construct_time[LAYER_NUMBER];
        int hist_const_finish_timepoint[LAYER_NUMBER][3];
        double idle_time[LAYER_NUMBER];
        double leaf_node_construct_time;
        int spawn_multi_thread_num=MAX_THREAD_NUMBER; 
        gettimeofday(&myt00, NULL);
        for (uint32_t tree = 1; tree <= TREE_NUMBER; tree++)
        {
            for(int ilayer=0;ilayer<LAYER_NUMBER;ilayer++)  layer_total_training_time[ilayer]=0.0;
            gettimeofday(&myt0, NULL);
            for (auto &p : node) p.clear();
            for (uint32_t i = 0; i < INSTANCE_NUMBER; i++) node[1].push_back(i);
            unique_g.clear();
            unique_h.clear();
            vector<double>::iterator gh_iter;
            int redo_layer=-1;
            for (uint32_t i = 0; i < INSTANCE_NUMBER; i++)
            {
                double predict_value = 0;
                for (uint32_t j = 0; j < tree; j++) predict_value += learning_rate * predict[j][i];
                predict_value = 1/(1+exp(-predict_value));
                g_inst[i] = first_order_gradient(inst_labels[i], predict_value);
                h_inst[i] = second_order_gradient(inst_labels[i], predict_value);

            }//for
            gettimeofday(&myt1, NULL);
            uint32_t i_layer;
            int remain_to_send,remain_to_receive,topN_feature_num,have_feat_num,pre_ele_ahead_num,node_num_startID_curl;
            int *int_rcv_pointer;
            double step2_feat_frac,step1_inst_frac;
            double *double_rcv_pointer;
            vector<vector<int>> receive_total_topN_feature;
            vector<vector<int>> history_feature;
            string send_s_local,pre_ele_result_str,pre_ele_ahead,schedule_info,s;
            int signal_finish_ahead;
            thread thread_hist[40];

            for(i_layer = 1; i_layer<=LAYER_NUMBER-1; i_layer++){
                gettimeofday(&myt1, NULL);
                schedule_info="";
                schedule_info=sock.recieve_specify(16);
                double_rcv_pointer=(double*)(schedule_info.c_str());
                step1_inst_frac=*double_rcv_pointer;
                step2_feat_frac=*(double_rcv_pointer+1);
                if(i_layer==LAYER_NUMBER-1)
                    step1_inst_frac=1;
                node_num_startID_curl=int(pow(2.0,i_layer-1));
                thread_bucket_g.resize(node_num_startID_curl*spawn_multi_thread_num,vector<vector<Binary_Double>>(FEATURE_NUMBER,vector<Binary_Double>(QUANTILE_NUMBER)));
                thread_bucket_h.resize(node_num_startID_curl*spawn_multi_thread_num,vector<vector<Binary_Double>>(FEATURE_NUMBER,vector<Binary_Double>(QUANTILE_NUMBER)));
                for(int x=0;x<node_num_startID_curl*spawn_multi_thread_num;x++){
                    for(int y=0;y<FEATURE_NUMBER;y++){
                        int dim1_idx=x*FEATURE_NUMBER+y;
                        for(int z=0;z<QUANTILE_NUMBER;z++){
                            thread_bucket_g[x][y][z].value=0;
                            thread_bucket_h[x][y][z].value=0;
                        }//for
                    }//for
                }//for
                hist_comp_thread_status.clear();         
                for(int id_thread=0;id_thread<spawn_multi_thread_num;id_thread++){
                    hist_comp_thread_status.push_back(0);
                    thread_hist[id_thread]=thread(thread_hist_const_new_pre_ele_first_sparse,node_num_startID_curl,id_thread,spawn_multi_thread_num,step1_inst_frac);
                }//for
                for(int batch_id=0;batch_id<(FEATURE_NUMBER-1)/pipeline_feature_batch+1;batch_id++){
                    for(int i_thread=0;i_thread<spawn_multi_thread_num;i_thread++){
                        while(1)
                            if(hist_comp_thread_status[i_thread]>=batch_id+1)   break;
                    }//for
                    string send_s_local;
                    int j_start=batch_id*pipeline_feature_batch;
                    int j_end=(batch_id+1)*pipeline_feature_batch;
                    if(j_end>FEATURE_NUMBER)    j_end=FEATURE_NUMBER;
                    for (uint32_t j = j_start; j <j_end; j++){
                        for(uint32_t i = 0; i < node_num_startID_curl; i++){
                            for(uint32_t q = 0; q < QUANTILE_NUMBER; q++){
                                for(int i_thread=1;i_thread<spawn_multi_thread_num;i_thread++){
                                    thread_bucket_g[0*node_num_startID_curl+i][j][q].value+=thread_bucket_g[i_thread*node_num_startID_curl+i][j][q].value;
                                    thread_bucket_h[0*node_num_startID_curl+i][j][q].value+=thread_bucket_h[i_thread*node_num_startID_curl+i][j][q].value;
                                }//for
                                for(int byte_c=0;byte_c<8;byte_c++){
                                    send_s_local+=thread_bucket_g[0*node_num_startID_curl+i][j][q].s[byte_c];
                                }//for
                                for(int byte_c=0;byte_c<8;byte_c++){
                                    send_s_local+=thread_bucket_h[0*node_num_startID_curl+i][j][q].s[byte_c];
                                }//for
                            }//for
                        }//for
                    }//for 
                    int remain_to_send=send_s_local.size();
                    while(remain_to_send>0){
                        if(remain_to_send>4194304){
                            sock.send_specify(send_s_local.substr(send_s_local.size()-remain_to_send,4194304),4194304);
                            remain_to_send-=4194304;
                        }//if
                        else{
                            sock.send_specify(send_s_local.substr(send_s_local.size()-remain_to_send,remain_to_send),remain_to_send);
                            remain_to_send=0;
                        }//if
                    }//while
                }//for
                for(int id_thread=0;id_thread<spawn_multi_thread_num;id_thread++){
                    thread_hist[id_thread].join();
                }//for
                gettimeofday(&myt5, NULL);
                if(i_layer==LAYER_NUMBER-1)
                    break;
                for(int x=0;x<node_num_startID_curl*spawn_multi_thread_num;x++){
                    for(int y=0;y<FEATURE_NUMBER;y++){
                        for(int z=0;z<QUANTILE_NUMBER;z++){
                            thread_bucket_g[x][y][z].value=0;
                            thread_bucket_h[x][y][z].value=0;
                        }//for
                    }//for
                }//for
                pre_ele_ahead_num=sock.recieve_uint();
                history_feature.resize(node_num_startID_curl);
                for(int i_node=0;i_node<node_num_startID_curl;i_node++) history_feature[i_node].clear();
                list_of_all_topN_feature.resize(node_num_startID_curl);
                for(int i_node=0;i_node<node_num_startID_curl;i_node++) list_of_all_topN_feature[i_node].clear();
                have_feat_num=0;
                for(int ahead_id=0;ahead_id<pre_ele_ahead_num;ahead_id++){
                    int ahead_length=sock.recieve_uint();
                    pre_ele_ahead=sock.recieve_specify(ahead_length);
                    int_rcv_pointer=(int*)(pre_ele_ahead.c_str());
                    for(uint32_t i_node = 0; i_node < node_num_startID_curl; i_node++){
                        for(int j=0;j<ahead_length/(4*node_num_startID_curl);j++){
                            list_of_all_topN_feature[i_node].push_back(*(int_rcv_pointer+i_node*ahead_length/(4*node_num_startID_curl)+j));
                        }//for
                    }//for
                    have_feat_num+=ahead_length/(4*node_num_startID_curl);
                    if(have_feat_num>=least_compute_feat_num){
                        for(int id_thread=0;id_thread<spawn_multi_thread_num;id_thread++){
                            thread_hist[id_thread]=thread(thread_hist_const_new_pre_ele_second_sparse,node_num_startID_curl,id_thread,spawn_multi_thread_num, step1_inst_frac);
                        }//for
                        for(int id_thread=0;id_thread<spawn_multi_thread_num;id_thread++){
                            thread_hist[id_thread].join();
                        }//for
                        for(uint32_t i_node = 0; i_node < node_num_startID_curl; i_node++){
                            for(int j=0;j<have_feat_num;j++)
                                history_feature[i_node].push_back(list_of_all_topN_feature[i_node][j]);
                            list_of_all_topN_feature[i_node].clear();
                        }//for             
                        for(int x=0;x<node_num_startID_curl*spawn_multi_thread_num;x++){
                            for(int y=0;y<FEATURE_NUMBER;y++){
                                for(int z=0;z<QUANTILE_NUMBER;z++){
                                    continue;
                                }//for
                            }//for
                        }//for
                        have_feat_num=0;
                    }//if
                }//for
                receive_total_topN_feature.resize(node_num_startID_curl);
                list_of_all_topN_feature.resize(node_num_startID_curl);
                topN_feature_num=FEATURE_NUMBER*step2_feat_frac;
                remain_to_receive=4*node_num_startID_curl*topN_feature_num;
                pre_ele_result_str=sock.recieve_specify(remain_to_receive);
                int_rcv_pointer=(int*)(pre_ele_result_str.c_str());
                for(uint32_t i_node = 0; i_node < node_num_startID_curl; i_node++){
                    list_of_all_topN_feature[i_node].clear();
                    receive_total_topN_feature[i_node].clear();
                    for(int j=0;j<topN_feature_num;j++){
                        receive_total_topN_feature[i_node].push_back(*(int_rcv_pointer+topN_feature_num*i_node+j));

                        if(receive_total_topN_feature[i_node][j]>=FEATURE_NUMBER || receive_total_topN_feature[i_node][j]<0)
                            cout<<"there is a j of "<<receive_total_topN_feature[i_node][j]<<endl;
                    }//for
                    int i_history=0;
                    if(history_feature[i_node].size()==0) history_feature[i_node].push_back(-1);
                    for(int i_total=0;i_total<receive_total_topN_feature[i_node].size();i_total++){
                        while(history_feature[i_node][i_history]<receive_total_topN_feature[i_node][i_total]){
                            if(i_history+1>=history_feature[i_node].size())
                                break;
                            i_history++;
                        }//while
                        if(history_feature[i_node][i_history]!=receive_total_topN_feature[i_node][i_total])
                            list_of_all_topN_feature[i_node].push_back(receive_total_topN_feature[i_node][i_total]);
                    }//for
                }//for
                gettimeofday(&myt7, NULL);
                for(int id_thread=0;id_thread<spawn_multi_thread_num;id_thread++){
                    thread_hist[id_thread]=thread(thread_hist_const_new_pre_ele_second_sparse,node_num_startID_curl,id_thread,spawn_multi_thread_num,step1_inst_frac);
                }//for
                for(int id_thread=0;id_thread<spawn_multi_thread_num;id_thread++){
                    thread_hist[id_thread].join();
                }//for
                gettimeofday(&myt8, NULL);
                send_s_local="";
                for(uint32_t i = 0; i < node_num_startID_curl; i++){
                    for (auto &j:receive_total_topN_feature[i]){
                        for(uint32_t q = 0; q < QUANTILE_NUMBER; q++){
                            for(int i_thread=1;i_thread<spawn_multi_thread_num;i_thread++){
                                thread_bucket_g[0*node_num_startID_curl+i][j][q].value+=thread_bucket_g[i_thread*node_num_startID_curl+i][j][q].value;
                                thread_bucket_h[0*node_num_startID_curl+i][j][q].value+=thread_bucket_h[i_thread*node_num_startID_curl+i][j][q].value;
                            }//for
                            for(int byte_c=0;byte_c<8;byte_c++){
                                send_s_local+=thread_bucket_g[0*node_num_startID_curl+i][j][q].s[byte_c];
                            }//for
                            for(int byte_c=0;byte_c<8;byte_c++){
                                send_s_local+=thread_bucket_h[0*node_num_startID_curl+i][j][q].s[byte_c];
                            }//for
                        }//for
                    }//for
                }//for 
                remain_to_send=send_s_local.size();
                while(remain_to_send>0){
                    if(remain_to_send>4194304){
                        sock.send_specify(send_s_local.substr(send_s_local.size()-remain_to_send,4194304),4194304);
                        remain_to_send-=4194304;
                    }//if
                    else{
                        sock.send_specify(send_s_local.substr(send_s_local.size()-remain_to_send,remain_to_send),remain_to_send);
                        remain_to_send=0;
                    }//if
                }//while
                if(0){
                    redo_point: i_layer=redo_layer;
                    node_num_startID_curl=int(pow(2.0,i_layer-1));
                    for(int clear_node_id=node_num_startID_curl*2;clear_node_id<node.size();clear_node_id++)
                        node[clear_node_id].clear();
                }//if
                split_feature.resize(node_num_startID_curl);
                split_bkt.resize(node_num_startID_curl);
                split_result.resize(MAX_THREAD_NUMBER);
                vector<double> split_score(node_num_startID_curl);
                for(uint32_t i = 0; i < node_num_startID_curl; i++){
                    string s;
                    s = sock.recieve_str(20);
                    split_feature[i] = b2u(s.substr(0, 4));
                    split_bkt[i] = int(b2d(s.substr(4, 12)));
                    split_score[i] = b2d(s.substr(12, 20));
                    if(split_bkt[i]==QUANTILE_NUMBER-1)
                        split_point[tree][i+node_num_startID_curl]=make_pair(split_feature[i], 1000000);
                    else
                        split_point[tree][i+node_num_startID_curl]=make_pair(split_feature[i], quantiles[split_feature[i]][split_bkt[i]]);
                }//for
                gettimeofday(&myt9, NULL);
                idle_time[i_layer]=train_time;
                for(uint32_t i = 0; i < node_num_startID_curl; i++){
                    for(int j=0;j<MAX_THREAD_NUMBER;j++)    split_result[j].clear();
                    for(int id_thread=0;id_thread<MAX_THREAD_NUMBER;id_thread++){
                        thread_hist[id_thread]=thread(thread_node_split,node_num_startID_curl+i,node_num_startID_curl,id_thread,MAX_THREAD_NUMBER);
                    }//for
                    for(int id_thread=0;id_thread<MAX_THREAD_NUMBER;id_thread++)
                        thread_hist[id_thread].join();
                    int inst_pointer_in_node=0;
                    for(int j=0;j<MAX_THREAD_NUMBER;j++){
                        for(auto &sig_rl: split_result[j]){
                            if(sig_rl==0)
                                node[2*(i+node_num_startID_curl)].push_back(node[i+node_num_startID_curl][inst_pointer_in_node]);
                            else    
                                node[2*(i+node_num_startID_curl)+1].push_back(node[i+node_num_startID_curl][inst_pointer_in_node]);
                            inst_pointer_in_node++;
                        }//for
                    }//for
                }//for
                gettimeofday(&myt10, NULL);
            }//for 
            for(uint32_t i_layer = LAYER_NUMBER-1; i_layer<LAYER_NUMBER; i_layer++){
                thread thread_hist[40];
                int node_num_startID_curl=int(pow(2.0,i_layer-1));
                split_feature.resize(node_num_startID_curl);
                split_bkt.resize(node_num_startID_curl);
                split_result.resize(MAX_THREAD_NUMBER);
                vector<double> split_score(node_num_startID_curl);
                for(uint32_t i = 0; i < node_num_startID_curl; i++){
                    string s;
                    s = sock.recieve_str(20);
                    split_feature[i] = b2u(s.substr(0, 4));
                    split_bkt[i] = int(b2d(s.substr(4, 12)));
                    split_score[i] = b2d(s.substr(12, 20));
                    if(split_bkt[i]==QUANTILE_NUMBER-1)
                        split_point[tree][i+node_num_startID_curl]=make_pair(split_feature[i], 1000000);
                    else
                        split_point[tree][i+node_num_startID_curl]=make_pair(split_feature[i], quantiles[split_feature[i]][split_bkt[i]]);
                }//for 
                gettimeofday(&myt6, NULL);
                for(uint32_t i = 0; i < node_num_startID_curl; i++){
                    for(int j=0;j<MAX_THREAD_NUMBER;j++)    split_result[j].clear();
                    for(int id_thread=0;id_thread<MAX_THREAD_NUMBER;id_thread++){
                        thread_hist[id_thread]=thread(thread_node_split,node_num_startID_curl+i,node_num_startID_curl,id_thread,MAX_THREAD_NUMBER);
                    }//for
                    for(int id_thread=0;id_thread<MAX_THREAD_NUMBER;id_thread++)
                        thread_hist[id_thread].join();
                    int inst_pointer_in_node=0;
                    for(int j=0;j<MAX_THREAD_NUMBER;j++){
                        for(auto &sig_rl: split_result[j]){
                            if(sig_rl==0)
                                node[2*(i+node_num_startID_curl)].push_back(node[i+node_num_startID_curl][inst_pointer_in_node]);
                            else    
                                node[2*(i+node_num_startID_curl)+1].push_back(node[i+node_num_startID_curl][inst_pointer_in_node]);
                            inst_pointer_in_node++;
                        }//for
                    }//for
                }//for
                gettimeofday(&myt7, NULL);
                s = sock.recieve_str(1);
                if(s[0]=='1'){
                    s=sock.recieve_str(4);
                    redo_layer = b2u(s.substr(0, 4));
                    goto redo_point;
                }//if
            }//for
            for (uint32_t i = int(pow(2.0,LAYER_NUMBER-1)); i < int(pow(2.0,LAYER_NUMBER)); i++)
            {
                int my_flag_pos_label=0;
                int my_flag_neg_label=0;
                double sum_g = 0, sum_h = 0;
                for (const auto &p : node[i])
                {
                    sum_g += g_inst[p];
                    sum_h += h_inst[p];
                    if(inst_labels[p]==1)
                        my_flag_pos_label=1;
                    if(inst_labels[p]==0)
                        my_flag_neg_label=1;
                }//for
                sock.send_double(sum_g);
                sock.send_double(sum_h);
                sum_g=sock.recieve_double();
                sum_h=sock.recieve_double();
                for (const auto &p : node[i])
                {
                    predict[tree][p] = - sum_g / (sum_h + my_lambda);
                }//for
                split_point[tree][i] = make_pair(-1, - sum_g / (sum_h + my_lambda));
            }//for
            gettimeofday(&myt8, NULL);
        }
        gettimeofday(&myt01, NULL);
    }
    


    void test(int cur_tree_finish){
        vector<Instance> test_instances;
        vector<double> raw_predict_scores;
        ifstream in(test_file, ios::in);
        assert(in.is_open());
        feature_name = read_line(in);
        for (uint32_t i = 0; i < test_INSTANCE_NUMBER; i++)
        {
            vector<string> vct = read_line(in);
            assert(vct.size() == FEATURE_NUMBER + 2);
            Instance instance1;
            instance1.name = vct[0];
            instance1.label = stod(vct[FEATURE_NUMBER+1]);
            assert(stod(vct[FEATURE_NUMBER+1])==0 || stod(vct[FEATURE_NUMBER+1])==1);
            for (uint32_t j = 0; j < FEATURE_NUMBER; j++)
                instance1.value.push_back(stod(vct[j+1]));
            test_instances.push_back(instance1);
        }//for
        for(uint32_t i = 0; i < test_INSTANCE_NUMBER; i++){
            raw_predict_scores.push_back(0.0);
            for(uint32_t tree_id = 1; tree_id <= cur_tree_finish; tree_id++){
                int cur_node_id=1;
                while(1){
                    if(split_point[tree_id][cur_node_id].first == -1){
                        raw_predict_scores[i]+=(learning_rate * split_point[tree_id][cur_node_id].second);
                        break;
                    }//if
                    if(test_instances[i].value[split_point[tree_id][cur_node_id].first] <= split_point[tree_id][cur_node_id].second)
                        cur_node_id=(cur_node_id+cur_node_id);
                    else
                        cur_node_id=(cur_node_id+cur_node_id+1);
                }
            }//for
        }//for
        vector<int> pos_inst_id;
        vector<int> neg_inst_id;
        for(uint32_t i = 0; i < test_INSTANCE_NUMBER; i++){
            if(test_instances[i].label>0)    pos_inst_id.push_back(i);
            else    neg_inst_id.push_back(i);
        }//for
        double I_sum=0;
        for(uint32_t i = 0; i < pos_inst_id.size(); i++){
            for(uint32_t j = 0; j < neg_inst_id.size(); j++){
                if(abs(raw_predict_scores[pos_inst_id[i]] - raw_predict_scores[neg_inst_id[j]])<0.0001)
                    I_sum+=0.5;
                else if(raw_predict_scores[pos_inst_id[i]]>raw_predict_scores[neg_inst_id[j]])
                    I_sum+=1;
                else
                    I_sum+=0;
            }//for
        }//for
    }

};


int main(int argc, char** argv)
{
    assert(argc == 12);
    WORKER_NUMBER = stoi(argv[1]);
    port_in = 30000;
    port_out = 40007;
    file = argv[2];
    test_file=argv[3];
    key_file = argv[4];
    INSTANCE_NUMBER = stoul(argv[5]);
    FEATURE_NUMBER = stoul(argv[6]);
    test_INSTANCE_NUMBER=stoul(argv[7]);
    QUANTILE_NUMBER=stoul(argv[8]);
    TREE_NUMBER=stoul(argv[9]);
    LAYER_NUMBER=stoul(argv[10]);
    MAX_THREAD_NUMBER=stod(argv[11]);
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    char server_addr[]="172.23.0.1";
    char self_addr[]="172.23.12.127";
    sock.link(server_addr,self_addr);
    Worker worker;
    worker.load_quantiles();
    worker.init_fd_orig();
    worker.sync_loading_dataset_finish();
    worker.train_pre_election();
    sock.close();
    return 0;
}
