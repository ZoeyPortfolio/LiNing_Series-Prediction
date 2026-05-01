import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
import io
import os

warnings.filterwarnings('ignore')

app = FastAPI(title="李宁快闪店分析API")

# ==================== 全局变量存储模型和预训练数据 ====================
scaler = None
models = {}
train_features = ['年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数']
target_cols = ['金标Proportion', '荣耀Proportion', '国家队Proportion', '其他Proportion']

# 省份分数映射
province_score = {
    '上海市': 4, '北京市': 4, '广东省': 4,
    '江苏省': 3, '浙江省': 3, '四川省': 3, '湖北省': 3, '湖南省': 3,
    '河南省': 3, '安徽省': 3, '福建省': 3, '陕西省': 3, '重庆市': 3,
    '天津市': 3, '山东省': 3, '辽宁省': 3,
    '河北省': 2, '江西省': 2, '广西壮族自治区': 2, '云南省': 2,
    '贵州省': 2, '山西省': 2, '吉林省': 2, '黑龙江省': 2,
}

# 省份Tier映射
province_tier = {
    '上海市': '一线', '北京市': '一线', '广东省': '一线',
    '江苏省': '强二线', '浙江省': '强二线', '四川省': '强二线', '湖北省': '强二线', 
    '湖南省': '强二线', '河南省': '强二线', '安徽省': '强二线', '福建省': '强二线', 
    '陕西省': '强二线', '重庆市': '强二线', '天津市': '强二线', '山东省': '强二线', 
    '辽宁省': '强二线',
    '河北省': '二线', '江西省': '二线', '广西壮族自治区': '二线', '云南省': '二线',
    '贵州省': '二线', '山西省': '二线', '吉林省': '二线', '黑龙江省': '二线',
}


def load_and_merge_data(population_df, phone_df, age_df, gender_df, asset_df):
    """合并所有sheet并计算特征指标"""
    phone_columns = ['赢商项目ID'] + ['APPLE', 'HUAWEI', 'SAMSUNG']
    phone_df = phone_df[phone_columns]
    
    age_columns = ['赢商项目ID'] + ['19-24', '25-29']
    age_df = age_df[age_columns]
    
    gender_columns = ['赢商项目ID', '女性占比']
    gender_df = gender_df[gender_columns]
    
    asset_columns = ['赢商项目ID'] + ['超级富豪', '富豪', '中产']
    asset_df = asset_df[asset_columns]
    
    df = population_df.merge(phone_df, on='赢商项目ID', how='inner')
    df = df.merge(age_df, on='赢商项目ID', how='inner')
    df = df.merge(gender_df, on='赢商项目ID', how='inner')
    df = df.merge(asset_df, on='赢商项目ID', how='inner')
    
    df['年轻占比'] = df['19-24'] + df['25-29']
    df['高消费力_资产'] = df['超级富豪'] + df['富豪'] + df['中产']
    df['高消费力_手机'] = df['APPLE'] + df['HUAWEI'] + df['SAMSUNG']
    df['高消费力'] = (df['高消费力_资产'] + df['高消费力_手机']) / 2
    
    df['省份分数'] = df['省份'].map(province_score)
    return df


def calculate_series_ratio(sales_detail_df):
    """计算15个快闪店的系列销售占比"""
    series_mapping = {
        '李宁荣耀金标': '金标',
        '李宁荣耀': '荣耀',
        '国家队': '国家队',
        '其他系列': '其他'
    }
    sales_detail_df['系列'] = sales_detail_df['系列'].map(series_mapping)
    
    sales_detail_filtered = sales_detail_df[sales_detail_df['品类'] != '推广类'].copy()
    sales_detail_filtered = sales_detail_filtered[pd.notna(sales_detail_filtered['销售数量'])]
    sales_detail_filtered = sales_detail_filtered[sales_detail_filtered['销售数量'] > 0]
    
    summary = sales_detail_filtered.groupby(['店铺名称', '系列'])['销售数量'].sum().reset_index()
    total_by_store = summary.groupby('店铺名称')['销售数量'].sum().reset_index()
    total_by_store.columns = ['店铺名称', '总销售数量']
    summary = summary.merge(total_by_store, on='店铺名称')
    summary['销售占比'] = summary['销售数量'] / summary['总销售数量']
    
    category_ratio_wide = summary.pivot_table(
        index='店铺名称', columns='系列', values='销售占比', fill_value=0
    ).reset_index()
    
    for s in ['金标', '荣耀', '国家队', '其他']:
        if s not in category_ratio_wide.columns:
            category_ratio_wide[s] = 0
    
    category_ratio_wide = category_ratio_wide.rename(columns={
        '金标': '金标Proportion',
        '荣耀': '荣耀Proportion',
        '国家队': '国家队Proportion',
        '其他': '其他Proportion'
    })
    
    ratio_cols = ['金标Proportion', '荣耀Proportion', '国家队Proportion', '其他Proportion']
    row_sum = category_ratio_wide[ratio_cols].sum(axis=1)
    for col in ratio_cols:
        category_ratio_wide[col] = category_ratio_wide[col] / row_sum
    
    return category_ratio_wide


def get_flash_cluster_result(df, flash_mapping_df):
    """对15个快闪店进行聚类"""
    all_features_df = df[['李宁商场名称', '年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数']].copy()
    liNing_store_names = flash_mapping_df['李宁商场名称'].unique().tolist()
    matched_features = all_features_df[all_features_df['李宁商场名称'].isin(liNing_store_names)].copy()
    
    cluster_features = ['年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数']
    scaler_flash = StandardScaler()
    X_scaled = scaler_flash.fit_transform(matched_features[cluster_features])
    
    kmeans_flash = KMeans(n_clusters=3, random_state=42, n_init=10)
    matched_features['客群类型'] = kmeans_flash.fit_predict(X_scaled)
    
    name_mapping = {0: "一线城市标杆店", 1: "女性高消潜力店", 2: "年轻潮流主力店"}
    matched_features['客群类型名称'] = matched_features['客群类型'].map(name_mapping)
    
    feature_mapping = matched_features[['李宁商场名称', '年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数', '客群类型名称']].drop_duplicates()
    result_df = flash_mapping_df.merge(feature_mapping, on='李宁商场名称', how='left')
    
    output_columns = ['店铺名称', '年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数', '客群类型名称']
    return result_df[output_columns]


def get_all_malls_cluster(df):
    """对所有商业体进行聚类"""
    cluster_features = ['年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数']
    all_malls_df = df[['李宁商场名称', '城市', '省份', '赢商项目ID'] + cluster_features].copy()
    all_malls_df = all_malls_df.dropna(subset=cluster_features)
    
    scaler_all = StandardScaler()
    X_scaled_all = scaler_all.fit_transform(all_malls_df[cluster_features])
    kmeans_all = KMeans(n_clusters=3, random_state=42, n_init=10)
    all_malls_df['客群类型'] = kmeans_all.fit_predict(X_scaled_all)
    
    name_mapping = {0: "一线城市标杆店", 1: "女性高消潜力店", 2: "年轻潮流主力店"}
    all_malls_df['客群类型名称'] = all_malls_df['客群类型'].map(name_mapping)
    
    all_malls_df['省份Tier'] = all_malls_df['省份'].map(province_tier)
    
    output_columns = ['李宁商场名称', '城市', '省份', '省份Tier', '赢商项目ID', 
                      '年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数', 
                      '客群类型', '客群类型名称']
    return all_malls_df[output_columns]


# ==================== 端点1：所有商业体各系列占比预测 ====================
@app.post("/series-ratio-predict")
async def series_ratio_predict(
    population_file: str = Form(...),
    phone_file: str = Form(...),
    age_file: str = Form(...),
    gender_file: str = Form(...),
    asset_file: str = Form(...),
    sales_file: str = Form(...),
    flash_mapping_file: str = Form(...)
):
    """生成所有商业体_各系列占比预测.xlsx（与原始代码完全一致）"""
    try:
        # 读取上传的文件
        from io import BytesIO
        import requests
        
        def download_file(url: str) -> pd.DataFrame:
            response = requests.get(url)
            response.raise_for_status()
            return pd.read_excel(BytesIO(response.content))
        
        population_df = download_file(population_file)
        phone_df = download_file(phone_file)
        age_df = download_file(age_file)
        gender_df = download_file(gender_file)
        asset_df = download_file(asset_file)
        sales_detail = download_file(sales_file)
        flash_mapping = download_file(flash_mapping_file)
        
        # 数据预处理
        df = load_and_merge_data(population_df, phone_df, age_df, gender_df, asset_df)
        
        # 15个快闪店聚类
        flash_cluster = get_flash_cluster_result(df, flash_mapping[['店铺名称', '李宁商场名称']].drop_duplicates())
        
        # 系列占比计算
        category_ratio = calculate_series_ratio(sales_detail)
        
        # 合并
        final_df = category_ratio.merge(
            flash_cluster[['店铺名称', '年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数', '客群类型名称']],
            on='店铺名称',
            how='inner'
        )
        
        # 各客群类型的平均系列占比
        type_summary = final_df.groupby('客群类型名称')[['金标Proportion', '荣耀Proportion', '国家队Proportion', '其他Proportion']].mean().round(4)
        
        # 训练随机森林模型
        X_train_raw = final_df[train_features].values
        y_train = final_df[target_cols].values
        
        scaler_model = StandardScaler()
        X_train = scaler_model.fit_transform(X_train_raw)
        
        models_dict = {}
        for i, col in enumerate(target_cols):
            rf = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_split=2)
            rf.fit(X_train, y_train[:, i])
            models_dict[col] = rf
        
        # 所有商业体聚类
        all_malls_df = get_all_malls_cluster(df)
        
        # 预测
        df_predict_raw = all_malls_df[train_features + ['李宁商场名称', '城市']].copy()
        df_predict_raw = df_predict_raw.dropna(subset=train_features)
        X_predict = scaler_model.transform(df_predict_raw[train_features])
        
        for col in target_cols:
            df_predict_raw[col] = models_dict[col].predict(X_predict)
        
        # 归一化
        row_sum = df_predict_raw[target_cols].sum(axis=1)
        for col in target_cols:
            df_predict_raw[col] = df_predict_raw[col] / row_sum
        
        # 保存到Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_predict_raw.to_excel(writer, sheet_name='所有商业体预测', index=False)
            type_summary.to_excel(writer, sheet_name='各客群类型平均占比')
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=所有商业体_各系列占比预测.xlsx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 端点2：一线和新一线高潜力商业体TOP20 ====================
@app.post("/top20")
async def top20(
    population_file: str = Form(...),
    phone_file: str = Form(...),
    age_file: str = Form(...),
    gender_file: str = Form(...),
    asset_file: str = Form(...),
    sales_file: str = Form(...),
    flash_mapping_file: str = Form(...)
):
    """生成一线和新一线高潜力商业体TOP20.xlsx（与原始代码完全一致）"""
    try:
        from io import BytesIO
        import requests
        
        def download_file(url: str) -> pd.DataFrame:
            response = requests.get(url)
            response.raise_for_status()
            return pd.read_excel(BytesIO(response.content))
        
        population_df = download_file(population_file)
        phone_df = download_file(phone_file)
        age_df = download_file(age_file)
        gender_df = download_file(gender_file)
        asset_df = download_file(asset_file)
        sales_detail = download_file(sales_file)
        flash_mapping = download_file(flash_mapping_file)
        
        # 数据预处理
        df = load_and_merge_data(population_df, phone_df, age_df, gender_df, asset_df)
        
        # 15个快闪店聚类
        flash_cluster = get_flash_cluster_result(df, flash_mapping[['店铺名称', '李宁商场名称']].drop_duplicates())
        
        # 系列占比计算
        category_ratio = calculate_series_ratio(sales_detail)
        
        # 合并
        final_df = category_ratio.merge(
            flash_cluster[['店铺名称', '年轻占比', '女性占比', '高消费力', '3公里工作人口', '省份分数', '客群类型名称']],
            on='店铺名称',
            how='inner'
        )
        
        # 训练模型
        X_train_raw = final_df[train_features].values
        y_train = final_df[target_cols].values
        
        scaler_model = StandardScaler()
        X_train = scaler_model.fit_transform(X_train_raw)
        
        models_dict = {}
        for i, col in enumerate(target_cols):
            rf = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_split=2)
            rf.fit(X_train, y_train[:, i])
            models_dict[col] = rf
        
        # 所有商业体聚类
        all_malls_df = get_all_malls_cluster(df)
        
        # 预测
        df_predict_raw = all_malls_df[train_features + ['李宁商场名称', '城市']].copy()
        df_predict_raw = df_predict_raw.dropna(subset=train_features)
        X_predict = scaler_model.transform(df_predict_raw[train_features])
        
        for col in target_cols:
            df_predict_raw[col] = models_dict[col].predict(X_predict)
        
        row_sum = df_predict_raw[target_cols].sum(axis=1)
        for col in target_cols:
            df_predict_raw[col] = df_predict_raw[col] / row_sum
        
        # TOP20 计算
        tier1_cities = ['上海市', '北京市', '深圳市', '广州市']
        new_tier1_cities = ['成都市', '杭州市', '重庆市', '武汉市', '苏州市', '西安市', '南京市', 
                             '长沙市', '郑州市', '天津市', '合肥市', '青岛市', '东莞市', '宁波市']
        
        # 一线城市金标TOP20
        df_tier1 = df_predict_raw[df_predict_raw['城市'].isin(tier1_cities)].copy()
        top_gold_tier1 = df_tier1.nlargest(20, '金标Proportion')[
            ['李宁商场名称', '城市', '金标Proportion', '荣耀Proportion', '国家队Proportion', '其他Proportion']
        ].copy().reset_index(drop=True)
        
        # 新一线城市金标TOP20
        df_new_tier1 = df_predict_raw[df_predict_raw['城市'].isin(new_tier1_cities)].copy()
        top_gold_new_tier1 = df_new_tier1.nlargest(20, '金标Proportion')[
            ['李宁商场名称', '城市', '金标Proportion', '荣耀Proportion', '国家队Proportion', '其他Proportion']
        ].copy().reset_index(drop=True)
        
        # 一线城市荣耀TOP20
        top_glory_tier1 = df_tier1.nlargest(20, '荣耀Proportion')[
            ['李宁商场名称', '城市', '金标Proportion', '荣耀Proportion', '国家队Proportion', '其他Proportion']
        ].copy().reset_index(drop=True)
        
        # 新一线城市荣耀TOP20
        top_glory_new_tier1 = df_new_tier1.nlargest(20, '荣耀Proportion')[
            ['李宁商场名称', '城市', '金标Proportion', '荣耀Proportion', '国家队Proportion', '其他Proportion']
        ].copy().reset_index(drop=True)
        
        # 一线城市国家队TOP20
        top_national_tier1 = df_tier1.nlargest(20, '国家队Proportion')[
            ['李宁商场名称', '城市', '金标Proportion', '荣耀Proportion', '国家队Proportion', '其他Proportion']
        ].copy().reset_index(drop=True)
        
        # 新一线城市国家队TOP20
        top_national_new_tier1 = df_new_tier1.nlargest(20, '国家队Proportion')[
            ['李宁商场名称', '城市', '金标Proportion', '荣耀Proportion', '国家队Proportion', '其他Proportion']
        ].copy().reset_index(drop=True)
        
        # 城市汇总
        df_all_filtered = df_predict_raw[df_predict_raw['城市'].isin(tier1_cities + new_tier1_cities)].copy()
        city_summary = df_all_filtered.groupby('城市').agg({
            '金标Proportion': 'mean',
            '荣耀Proportion': 'mean',
            '国家队Proportion': 'mean',
            '李宁商场名称': 'count'
        }).rename(columns={'李宁商场名称': '商业体数量'}).round(4)
        
        # 保存到Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            top_gold_tier1.to_excel(writer, sheet_name='一线城市_金标TOP20', index=False)
            top_gold_new_tier1.to_excel(writer, sheet_name='新一线城市_金标TOP20', index=False)
            top_glory_tier1.to_excel(writer, sheet_name='一线城市_荣耀TOP20', index=False)
            top_glory_new_tier1.to_excel(writer, sheet_name='新一线城市_荣耀TOP20', index=False)
            top_national_tier1.to_excel(writer, sheet_name='一线城市_国家队TOP20', index=False)
            top_national_new_tier1.to_excel(writer, sheet_name='新一线城市_国家队TOP20', index=False)
            city_summary.to_excel(writer, sheet_name='城市汇总')
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=一线和新一线高潜力商业体TOP20.xlsx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)