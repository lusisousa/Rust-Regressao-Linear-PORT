# Data Structure Strategy and Implementation (Rust)

Este projeto foi desenvolvido como parte da disciplina **Data Structure Strategy and Implementation**, utilizando a linguagem **Rust**.  
O objetivo é demonstrar a implementação prática de estruturas e estratégias computacionais aplicadas a problemas reais, com foco em desempenho, segurança de memória e testes automatizados.

---

## 🚀 Funcionalidades Implementadas

### 🔹 Regressão Linear
Foi implementado um módulo de **regressão linear simples**, capaz de ajustar um modelo linear a partir de um conjunto de dados e calcular métricas estatísticas associadas.

Funções principais:
- `fit_and_predict(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64>`  
  Ajusta o modelo linear e retorna os valores previstos.

- `mse(y: &Vec<f64>, y_pred: &Vec<f64>) -> f64`  
  Calcula o **Erro Quadrático Médio (MSE)** entre valores reais e previstos.

- `r2_score(y: &Vec<f64>, y_pred: &Vec<f64>) -> f64`  
  Calcula o **Coeficiente de Determinação (R²)**, que mede a qualidade do ajuste.

- `variance_zero_case()`  
  Testa cenários de variância nula nos dados de entrada.

---

## 🧪 Testes Automatizados
O projeto contém **testes unitários** para garantir a corretude das implementações.  
Os testes cobrem:
- Ajuste e previsão (`fit_and_predict`)  
- Cálculo do MSE  
- Cálculo do R²  
- Casos com variância zero  
- Casos com R² indefinido  

Rodando os testes:
```bash
cargo test
```

---

## 📂 Estrutura do Projeto
```
src/
 ├── lib.rs              # Ponto de entrada da biblioteca
 ├── linear_regression.rs # Implementação da regressão linear
tests/
 ├── linear_regression.rs # Testes unitários
```

---

## ⚡ Tecnologias
- **Rust** (linguagem principal)  
- **Cargo** (gerenciador de pacotes e testes)  

---

## 👨‍💻 Autor
Projeto desenvolvido para fins acadêmicos na disciplina **Data Structure Strategy and Implementation**.
