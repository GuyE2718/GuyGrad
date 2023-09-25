#ifndef GUYGRAD_H
#define GUYGRAD_H

#include <functional>
#include <unordered_set>
#include <string>
#include <memory>

class Value : public std::enable_shared_from_this<Value> {
public:
    Value(float data, std::unordered_set<std::shared_ptr<Value>> prev = {}, std::string op = "");

    void set_grad(float grad_value);
    float get_data();
    void set_data(float data);
    float get_grad() const;
    std::unordered_set<std::shared_ptr<Value>>  get_prev() const;
    std::shared_ptr<Value> tanh();
    std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator-();
    std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> pow(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& other);

    void backward();

    float data;
    float grad;
    std::function<void()> _backward;
    std::unordered_set<std::shared_ptr<Value>> prev;
    std::string op;
};

#endif  
