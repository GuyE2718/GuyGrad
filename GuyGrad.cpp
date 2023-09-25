#include <functional>
#include <unordered_set>
#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include "GuyGrad.h"



Value::Value(float data, std::unordered_set<std::shared_ptr<Value>> prev, std::string op) {
    this->data = data;
    this->grad = 0.0;
    this->prev = std::move(prev);
    this->op = std::move(op);
    this->_backward = [this] {
        for (const auto& child : this->prev) {
            child->_backward();
        }
    };
}

float Value::get_data() {
    return data;
}

void Value::set_data(float data) {
    this->data = data;
}

std::unordered_set<std::shared_ptr<Value>> Value::get_prev() const {
    return prev;
}

float Value::get_grad() const {
    return grad;
}

void Value::set_grad(float grad_value) {
    this->grad = grad_value;
}

std::shared_ptr<Value> Value::tanh() {
    auto out = std::make_shared<Value>(std::tanh(data), std::unordered_set<std::shared_ptr<Value>>{ shared_from_this() }, "tanh");

    out->_backward = [this, out] {
        grad += (1 - std::pow(std::tanh(data), 2)) * out->grad; 
    };

    return out;
}


std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value>& other) {
    auto out_prev = std::unordered_set<std::shared_ptr<Value>>{ shared_from_this(), other };

    auto out = std::make_shared<Value>(data + other->data, out_prev, "+");

    out->_backward = [this, other, out] {
        grad += out->grad;
        other->grad += out->grad;

    };
    return out;
}


std::shared_ptr<Value> Value::operator-() {
    auto neg_one = std::make_shared<Value>(-1.0f); 
    return (*this) * neg_one; 
}


std::shared_ptr<Value> Value::operator-(const std::shared_ptr<Value>& other) {
    auto minusOne = std::make_shared<Value>(-1.0); 
    auto out_prev = std::unordered_set<std::shared_ptr<Value>>{ shared_from_this(), minusOne };
    auto out = std::make_shared<Value>(data * -1.0, out_prev, "*");
    out->_backward = [this, minusOne, out] {
        grad += minusOne->data * out->grad;
        minusOne->grad += data * out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::pow(const std::shared_ptr<Value>& other) {
    auto out_prev = std::unordered_set<std::shared_ptr<Value>>{ shared_from_this(), other };

    auto out = std::make_shared<Value>(std::pow(data, other->data), out_prev, "^");

    out->_backward = [this, other, out] {
        grad += other->data * std::pow(data, other->data - 1) * out->grad;

    };
    return out;
}

std::shared_ptr<Value> Value::operator/(const std::shared_ptr<Value>& other) {
    auto inverse = std::make_shared<Value>(1.0f / other->data); 
    auto out_prev = std::unordered_set<std::shared_ptr<Value>>{ shared_from_this(), inverse };
    auto out = std::make_shared<Value>(data * inverse->data, out_prev, "/");
    out->_backward = [this, other, inverse, out] {
        grad += inverse->data * out->grad;
        other->grad += -1.0f * (data / (other->data * other->data)) * out->grad; 
    };
    return out;
}

std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value>& other) {
    auto out_prev = std::unordered_set<std::shared_ptr<Value>>{ shared_from_this(), other };

    auto out = std::make_shared<Value>(data * other->data, out_prev, "*");

    out->_backward = [this, other, out] {
        grad += other->data * out->grad;
        other->grad += data * out->grad;

    };
    return out;
}


void Value::backward() {
    std::vector<std::shared_ptr<Value>> topo;
    std::unordered_set<std::shared_ptr<Value>> visited;

    std::function<void(const std::shared_ptr<Value>&)> build_topo = [&](const std::shared_ptr<Value>& v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);

            for (const auto& child : v->prev) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    build_topo(shared_from_this());

    grad = 1.0;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const auto& v = *it;
        v->_backward();
    }
}

float data;
float grad;
std::function<void()> _backward;
std::unordered_set<std::shared_ptr<Value>> prev;
std::string op;


std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return (*a) + b;
}
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return (*a) - b;
}
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return (*a) * b;
}
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return (*a) / b;
}



int main() {
    auto x1 = std::make_shared<Value>(2.0);
    auto x2 = std::make_shared<Value>(0.0);

    auto w1 = std::make_shared<Value>(-3.0);
    auto w2 = std::make_shared<Value>(1.0);

    auto b = std::make_shared<Value>(6.8813735870195432);

    auto x1w1 = *x1 * w1;
    auto x2w2 = *x2 * w2;

    auto x1w1x2w2 = *x1w1 + x2w2;
    auto n = *x1w1x2w2 + b;

    auto o = n->tanh();

    o->backward();

    

    std::cout << "w1 grad: " << w1->grad << std::endl;
    std::cout << "w2 grad: " << w2->grad << std::endl;
    
}
