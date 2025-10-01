import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf


def tickers_df(stocks):
    # Crear Diccionario de tickers y dataframes
    tickers = {}
    financials = {}

    for stock in stocks:
        tickers[stock] = yf.Ticker(stock)
        financials[stock] = tickers[stock].financials
        

    return tickers, financials

def historical(stocks, inicio, final):
    # Datos históricos
    historical_data = {}

    for stock in stocks:
        # Descargar individualmente para evitar MultiIndex
        data = yf.download(stock, start=inicio, end=final)
        
        # Si las columnas tienen MultiIndex, aplanarlas
        if isinstance(data.columns, pd.MultiIndex):
            # Tomar solo el primer nivel de las columnas (eliminar el nombre del stock)
            data.columns = data.columns.get_level_values(0)
        
        historical_data[stock] = data
        
        # Limpiar índice si es necesario
        if historical_data[stock].index.name is not None and "Ticker" in historical_data[stock].index:
            historical_data[stock] = historical_data[stock].drop(index="Ticker")
    
    return historical_data

def faltantes(historical_data, stocks):
    historical_data_completo = {}
    
    for stock in stocks:
        datee = historical_data[stock]
        # Crear línea del tiempo completa (todos los días hábiles)
        fechas_completas = pd.date_range(start=datee.index.min(), end=datee.index.max(), freq='B')

        # Fechas realmente presentes
        fechas_presentes = pd.to_datetime(datee.index)

        # Fechas faltantes
        fechas_faltantes = fechas_completas.difference(fechas_presentes)

        # Crear DataFrame completo con todas las fechas
        df_completo = pd.DataFrame(index=fechas_completas)
        
        # Rellenar con datos existentes y dejar NaN para fechas faltantes
        for columna in datee.columns:
            # Asegurar que las columnas mantengan nombres simples
            col_name = columna[0] if isinstance(columna, tuple) else columna
            df_completo[col_name] = datee[columna]
        
        # Almacenar el DataFrame completo
        historical_data_completo[stock] = df_completo

        print("Número de días faltantes en", stock, ":", len(fechas_faltantes))
        '''print("Fechas faltantes en la línea del tiempo:")
        print(fechas_faltantes)'''

        fig, ax = plt.subplots(figsize=(12, 2))
        ax.scatter(fechas_faltantes, np.ones(len(fechas_faltantes)), color='red', marker='|', s=200, label="Día faltante")

        # Líneas grandes para cada año (incluyendo el primero y el siguiente al último)
        year_start = fechas_completas.min().year
        year_end = fechas_completas.max().year + 1  # Para mostrar el siguiente año al último
        for year in range(year_start, year_end + 1):
            ax.axvline(pd.Timestamp(f"{year}-01-01"), color='black', linestyle='-', linewidth=1.5, alpha=0.7)
            ax.text(pd.Timestamp(f"{year}-01-01"), 1.05, str(year), ha='center', va='bottom', fontsize=9, color='black')

        # Líneas delgadas para cada mes
        for year in range(year_start, year_end):
            for month in range(1, 13):
                ax.axvline(pd.Timestamp(f"{year}-{month:02d}-01"), color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

        # Línea verde para el inicio real de los datos
        ax.axvline(datee.index.min(), color='green', linestyle='--', linewidth=2, label='Inicio real')
        ax.text(datee.index.min(), 1.1, f"Inicio datos\n{datee.index.min().date()}", color='green', ha='left', va='bottom', fontsize=9)

        # Línea roja para el final real de los datos
        ax.axvline(datee.index.max(), color='red', linestyle='--', linewidth=2, label='Fin real')
        ax.text(datee.index.max(), 1.1, f"Fin datos\n{datee.index.max().date()}", color='red', ha='right', va='bottom', fontsize=9)

        ax.set_xlim(fechas_completas.min(), pd.Timestamp(f"{year_end}-01-01"))
        ax.set_yticks([])
        ax.set_xlabel('Fecha')
        ax.set_title(f"Días faltantes en: {stock}")
        plt.tight_layout()
        plt.show()
    
    return historical_data_completo

def plot_stocks_price(historical_data, stocks, columna):
    fig, ax = plt.subplots(figsize=(12, 6))
    for stock in stocks:
        historical_data[stock][columna].plot(ax=ax, label=stock)

    # Formato del eje x: años y meses
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter(''))

    # Líneas grandes para cada año
    for year_tick in ax.xaxis.get_majorticklocs():
        ax.axvline(x=year_tick, color='black', linestyle='-', linewidth=1, alpha=0.7)

    # Líneas delgadas para cada mes
    for month_tick in ax.xaxis.get_minorticklocs():
        ax.axvline(x=month_tick, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)


    fechas = historical_data[stocks[0]].index  # O usa el índice de cualquier DataFrame de tu diccionario
    ax.set_xlim(fechas.min(), fechas.max())

    ax.set_title(f"{columna} para todos los stocks")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(columna)
    ax.legend()
    plt.tight_layout()
    plt.show()




def plot_stock_all_categories(historical_data, stock):
    columnas = [col for col in historical_data[stock].columns if col != "Volume"]
    print([f"'{col}'" for col in historical_data[stock].columns if col != "Volume"])

    # Primera gráfica: todas las categorías excepto Volume
    fig, ax = plt.subplots(figsize=(12, 6))
    for columna in columnas:
        historical_data[stock][columna].plot(ax=ax, label=columna)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter(''))

    for year_tick in ax.xaxis.get_majorticklocs():
        ax.axvline(x=year_tick, color='black', linestyle='-', linewidth=1, alpha=0.7)
    for month_tick in ax.xaxis.get_minorticklocs():
        ax.axvline(x=month_tick, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    fechas = historical_data[stock].index
    ax.set_xlim(fechas.min(), fechas.max())

    ax.set_title(f"Todas las categorías (excepto Volume) para {stock}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Segunda gráfica: solo Volume
    if "Volume" in historical_data[stock].columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        historical_data[stock]["Volume"].plot(ax=ax, color='purple', label="Volume")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_formatter(mdates.DateFormatter(''))
        for year_tick in ax.xaxis.get_majorticklocs():
            ax.axvline(x=year_tick, color='black', linestyle='-', linewidth=1, alpha=0.7)
        for month_tick in ax.xaxis.get_minorticklocs():
            ax.axvline(x=month_tick, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.set_xlim(fechas.min(), fechas.max())
        ax.set_title(f"Volume para {stock}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Volume")
        ax.legend()
        plt.tight_layout()
        plt.show()