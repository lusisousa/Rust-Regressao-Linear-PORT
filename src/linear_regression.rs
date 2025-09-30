#[derive(Debug, Clone, PartialEq)]
pub struct LinearRegression {
    pub slope: f64,
    pub intercept: f64,
}

impl LinearRegression {
    // Ajusta a regressão linear com base em vetores x e y
    pub fn fit_xy(x: &[f64], y: &[f64]) -> Result<Self, String> {
        if x.len() != y.len() {
            return Err("x e y devem ter o mesmo tamanho".into());
        }
        let n = x.len();
        if n == 0 {
            return Err("entrada vazia".into());
        }

        let n_f = n as f64;
        let soma_x = x.iter().copied().sum::<f64>();
        let soma_y = y.iter().copied().sum::<f64>();
        let soma_xy = x.iter().zip(y.iter()).map(|(xi, yi)| (*xi) * (*yi)).sum::<f64>();
        let soma_x2 = x.iter().map(|xi| (*xi) * (*xi)).sum::<f64>();

        let denom = n_f * soma_x2 - soma_x * soma_x;
        if denom.abs() < 1e-12 {
            return Err("variância de x é zero - não é possível calcular a inclinação".into());
        }

        let slope = (n_f * soma_xy - soma_x * soma_y) / denom;
        let intercept = (soma_y - slope * soma_x) / n_f;
        Ok(LinearRegression { slope, intercept })
    }

    // Ajusta usando apenas y, assumindo x = 0..n-1
    pub fn fit_series(y: &[f64]) -> Result<Self, String> {
        let n = y.len();
        if n == 0 {
            return Err("entrada vazia".into());
        }
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        Self::fit_xy(&x, y)
    }

    // Faz previsão para um único valor de x
    pub fn predict(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }

    // Faz previsões para vários valores de x
    pub fn predict_many(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&xi| self.predict(xi)).collect()
    }

    // Calcula o Erro Quadrático Médio
    pub fn mse_xy(&self, x: &[f64], y: &[f64]) -> Result<f64, String> {
        if x.len() != y.len() {
            return Err("x e y devem ter o mesmo tamanho".into());
        }
        let n = x.len();
        if n == 0 {
            return Err("entrada vazia".into());
        }
        let mse = x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| {
                let pred = self.predict(*xi);
                (*yi - pred).powi(2)
            })
            .sum::<f64>() / (n as f64);
        Ok(mse)
    }

    // Calcula o Coeficiente de Determinação
    pub fn r2_xy(&self, x: &[f64], y: &[f64]) -> Result<f64, String> {
        if x.len() != y.len() {
            return Err("x e y devem ter o mesmo tamanho".into());
        }
        let n = x.len();
        if n == 0 {
            return Err("entrada vazia".into());
        }
        let media_y = y.iter().copied().sum::<f64>() / (n as f64);
        let ss_tot = y.iter().map(|yi| (*yi - media_y).powi(2)).sum::<f64>();
        if ss_tot.abs() < 1e-12 {
            return Err("variância de y é zero - R² indefinido".into());
        }
        let ss_res = x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| {
                let pred = self.predict(*xi);
                (*yi - pred).powi(2)
            })
            .sum::<f64>();
        Ok(1.0 - ss_res / ss_tot)
    }

    // Faz previsão dos próximos k pontos de uma série temporal
    pub fn forecast_from_series(y: &[f64], k: usize) -> Result<Vec<f64>, String> {
        let modelo = Self::fit_series(y)?;
        let inicio = y.len() as f64;
        let preds = (0..k)
            .map(|i| modelo.predict(inicio + i as f64))
            .collect::<Vec<f64>>();
        Ok(preds)
    }
}


// Testes Unitarios //

#[cfg(test)]
mod tests {
    use super::*;

    fn quase_igual(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn teste_ajuste_e_previsao() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let modelo = LinearRegression::fit_xy(&x, &y).unwrap();
        assert!(quase_igual(modelo.slope, 2.0, 1e-12));
        assert!(quase_igual(modelo.intercept, 0.0, 1e-12));
        assert!(quase_igual(modelo.predict(5.0), 10.0, 1e-12));
    }

    #[test]
    fn teste_mse_e_r2() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let modelo = LinearRegression::fit_xy(&x, &y).unwrap();
        assert!(modelo.mse_xy(&x, &y).unwrap() < 1e-12);
        assert!(quase_igual(modelo.r2_xy(&x, &y).unwrap(), 1.0, 1e-12));
    }

    #[test]
    fn teste_forecast() {
        let y = vec![1.0, 2.0, 3.0];
        let preds = LinearRegression::forecast_from_series(&y, 2).unwrap();
        assert!(quase_igual(preds[0], 4.0, 1e-12));
        assert!(quase_igual(preds[1], 5.0, 1e-12));
    }
}