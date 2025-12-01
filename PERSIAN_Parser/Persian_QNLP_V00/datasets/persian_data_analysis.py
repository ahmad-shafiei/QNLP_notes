#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تحلیل جامع داده‌های فارسی
Comprehensive Persian Data Analysis Script

این اسکریپت تحلیل کاملی از داده‌های متنی فارسی ارائه می‌دهد
This script provides comprehensive analysis of Persian text data
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import json
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# تنظیم فونت فارسی برای matplotlib
plt.rcParams['font.family'] = ['Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PersianDataAnalyzer:
    def __init__(self, data_directory="."):
        """
        کلاس تحلیل داده‌های فارسی
        Persian Data Analyzer Class
        """
        self.data_directory = data_directory
        self.datasets = {}
        self.analysis_results = {}
        self.persian_stopwords = {
            'را', 'به', 'از', 'در', 'با', 'که', 'این', 'آن', 'و', 'یا', 'تا', 
            'برای', 'روی', 'زیر', 'کنار', 'پیش', 'نزد', 'میان', 'بین'
        }
        
    def load_data(self):
        """بارگذاری داده‌ها از فایل‌های متنی"""
        print("🔄 در حال بارگذاری داده‌ها...")
        
        file_patterns = {
            'train': 'mc_train_data.txt',
            'dev': 'mc_dev_data.txt', 
            'test': 'mc_test_data.txt'
        }
        
        for dataset_name, filename in file_patterns.items():
            filepath = os.path.join(self.data_directory, filename)
            if os.path.exists(filepath):
                data = []
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split(' ', 1)
                            if len(parts) == 2:
                                label = int(parts[0])
                                text = parts[1]
                                data.append({'label': label, 'text': text})
                
                self.datasets[dataset_name] = pd.DataFrame(data)
                print(f"✅ {dataset_name}: {len(data)} نمونه بارگذاری شد")
            else:
                print(f"❌ فایل {filename} یافت نشد")
    
    def preprocess_text(self, text):
        """پیش‌پردازش متن فارسی"""
        # حذف علائم نگارشی
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        # نرمال‌سازی فاصله‌ها
        text = re.sub(r'\s+', ' ', text)
        # حذف فاصله‌های ابتدا و انتها
        text = text.strip()
        return text
    
    def extract_words(self, text):
        """استخراج کلمات از متن"""
        text = self.preprocess_text(text)
        words = text.split()
        # حذف stop words
        words = [word for word in words if word not in self.persian_stopwords]
        return words
    
    def basic_statistics(self):
        """محاسبه آمار پایه"""
        print("\n📊 در حال محاسبه آمار پایه...")
        
        stats = {}
        for dataset_name, df in self.datasets.items():
            dataset_stats = {}
            
            # آمار کلی
            dataset_stats['total_samples'] = len(df)
            dataset_stats['label_distribution'] = df['label'].value_counts().to_dict()
            dataset_stats['label_percentage'] = (df['label'].value_counts(normalize=True) * 100).to_dict()
            
            # آمار متن
            df['text_length'] = df['text'].str.len()
            df['word_count'] = df['text'].apply(lambda x: len(self.extract_words(x)))
            
            dataset_stats['text_length'] = {
                'mean': df['text_length'].mean(),
                'median': df['text_length'].median(),
                'std': df['text_length'].std(),
                'min': df['text_length'].min(),
                'max': df['text_length'].max()
            }
            
            dataset_stats['word_count'] = {
                'mean': df['word_count'].mean(),
                'median': df['word_count'].median(),
                'std': df['word_count'].std(),
                'min': df['word_count'].min(),
                'max': df['word_count'].max()
            }
            
            stats[dataset_name] = dataset_stats
        
        self.analysis_results['basic_stats'] = stats
        return stats
    
    def analyze_vocabulary(self):
        """تحلیل واژگان"""
        print("\n📚 در حال تحلیل واژگان...")
        
        vocab_analysis = {}
        
        for dataset_name, df in self.datasets.items():
            all_words = []
            words_by_label = {0: [], 1: []}
            
            for _, row in df.iterrows():
                words = self.extract_words(row['text'])
                all_words.extend(words)
                words_by_label[row['label']].extend(words)
            
            vocab_stats = {}
            vocab_stats['total_words'] = len(all_words)
            vocab_stats['unique_words'] = len(set(all_words))
            vocab_stats['vocabulary_richness'] = len(set(all_words)) / len(all_words) if all_words else 0
            
            # کلمات پرتکرار
            word_freq = Counter(all_words)
            vocab_stats['most_common_words'] = word_freq.most_common(20)
            
            # تحلیل کلمات بر اساس برچسب
            label_vocab = {}
            for label in [0, 1]:
                label_words = words_by_label[label]
                if label_words:
                    label_freq = Counter(label_words)
                    label_vocab[label] = {
                        'total_words': len(label_words),
                        'unique_words': len(set(label_words)),
                        'most_common': label_freq.most_common(10)
                    }
            
            vocab_stats['label_vocabulary'] = label_vocab
            vocab_analysis[dataset_name] = vocab_stats
        
        self.analysis_results['vocabulary'] = vocab_analysis
        return vocab_analysis
    
    def pattern_analysis(self):
        """تحلیل الگوهای متنی"""
        print("\n🔍 در حال تحلیل الگوهای متنی...")
        
        patterns = {}
        
        # الگوهای مهم برای تشخیص کلاس
        cooking_patterns = ['پخت', 'آماده', 'درست', 'سس', 'غذا', 'شام', 'خوشمزه', 'ماهر']
        tech_patterns = ['برنامه', 'نرم افزار', 'اپ', 'اجرا', 'اشکالزدایی', 'آماده']
        
        for dataset_name, df in self.datasets.items():
            pattern_stats = {}
            
            # شمارش الگوهای آشپزی
            cooking_count = df['text'].str.contains('|'.join(cooking_patterns)).sum()
            tech_count = df['text'].str.contains('|'.join(tech_patterns)).sum()
            
            pattern_stats['cooking_mentions'] = cooking_count
            pattern_stats['tech_mentions'] = tech_count
            
            # تحلیل الگو بر اساس برچسب
            label_patterns = {}
            for label in [0, 1]:
                label_df = df[df['label'] == label]
                label_cooking = label_df['text'].str.contains('|'.join(cooking_patterns)).sum()
                label_tech = label_df['text'].str.contains('|'.join(tech_patterns)).sum()
                
                label_patterns[label] = {
                    'cooking_patterns': label_cooking,
                    'tech_patterns': label_tech,
                    'total_samples': len(label_df)
                }
            
            pattern_stats['by_label'] = label_patterns
            patterns[dataset_name] = pattern_stats
        
        self.analysis_results['patterns'] = patterns
        return patterns
    
    def create_visualizations(self):
        """ایجاد نمودارهای تحلیلی"""
        print("\n📈 در حال ایجاد نمودارها...")
        
        # ایجاد پوشه برای ذخیره نمودارها
        viz_dir = "analysis_visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. توزیع برچسب‌ها
        self._plot_label_distribution(viz_dir)
        
        # 2. توزیع طول متن
        self._plot_text_length_distribution(viz_dir)
        
        # 3. توزیع تعداد کلمات
        self._plot_word_count_distribution(viz_dir)
        
        # 4. کلمات پرتکرار
        self._plot_word_frequency(viz_dir)
        
        # 5. ابر کلمات
        self._create_wordclouds(viz_dir)
        
        # 6. مقایسه dataset ها
        self._plot_dataset_comparison(viz_dir)
        
        # 7. تحلیل الگوها
        self._plot_pattern_analysis(viz_dir)
        
        print(f"✅ نمودارها در پوشه {viz_dir} ذخیره شدند")
    
    def _plot_label_distribution(self, viz_dir):
        """نمودار توزیع برچسب‌ها"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (dataset_name, df) in enumerate(self.datasets.items()):
            label_counts = df['label'].value_counts()
            
            # نمودار دایره‌ای
            axes[i].pie(label_counts.values, labels=['کلاس 0', 'کلاس 1'], 
                       autopct='%1.1f%%', startangle=90)
            axes[i].set_title(f'توزیع برچسب‌ها - {dataset_name}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'label_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_text_length_distribution(self, viz_dir):
        """نمودار توزیع طول متن"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (dataset_name, df) in enumerate(self.datasets.items()):
            # هیستوگرام طول متن
            axes[i].hist(df['text_length'], bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'توزیع طول متن - {dataset_name}')
            axes[i].set_xlabel('طول متن (کاراکتر)')
            axes[i].set_ylabel('فراوانی')
        
        # نمودار مقایسه‌ای
        if len(self.datasets) > 1:
            for dataset_name, df in self.datasets.items():
                axes[3].hist(df['text_length'], alpha=0.5, label=dataset_name, bins=20)
            axes[3].set_title('مقایسه توزیع طول متن')
            axes[3].set_xlabel('طول متن (کاراکتر)')
            axes[3].set_ylabel('فراوانی')
            axes[3].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'text_length_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_word_count_distribution(self, viz_dir):
        """نمودار توزیع تعداد کلمات"""
        fig, axes = plt.subplots(1, len(self.datasets), figsize=(5*len(self.datasets), 5))
        if len(self.datasets) == 1:
            axes = [axes]
        
        for i, (dataset_name, df) in enumerate(self.datasets.items()):
            # Box plot برای تعداد کلمات بر اساس برچسب
            data_by_label = [df[df['label'] == label]['word_count'].values for label in [0, 1]]
            axes[i].boxplot(data_by_label, labels=['کلاس 0', 'کلاس 1'])
            axes[i].set_title(f'توزیع تعداد کلمات - {dataset_name}')
            axes[i].set_ylabel('تعداد کلمات')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'word_count_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_word_frequency(self, viz_dir):
        """نمودار فراوانی کلمات"""
        for dataset_name, vocab_data in self.analysis_results['vocabulary'].items():
            words, frequencies = zip(*vocab_data['most_common_words'][:15])
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(words)), frequencies)
            plt.xticks(range(len(words)), words, rotation=45)
            plt.title(f'کلمات پرتکرار - {dataset_name}')
            plt.ylabel('فراوانی')
            
            # افزودن مقادیر روی نمودار
            for bar, freq in zip(bars, frequencies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(freq), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'word_frequency_{dataset_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_wordclouds(self, viz_dir):
        """ایجاد ابر کلمات"""
        for dataset_name, df in self.datasets.items():
            # ابر کلمات کلی
            all_text = ' '.join(df['text'])
            processed_text = self.preprocess_text(all_text)
            
            if processed_text:
                wordcloud = WordCloud(width=800, height=400, 
                                     background_color='white',
                                     max_words=100,
                                     font_path='tahoma.ttf' if os.path.exists('tahoma.ttf') else None).generate(processed_text)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'ابر کلمات - {dataset_name}')
                plt.savefig(os.path.join(viz_dir, f'wordcloud_{dataset_name}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            # ابر کلمات بر اساس برچسب
            for label in [0, 1]:
                label_text = ' '.join(df[df['label'] == label]['text'])
                processed_text = self.preprocess_text(label_text)
                
                if processed_text:
                    wordcloud = WordCloud(width=800, height=400,
                                         background_color='white',
                                         max_words=50,
                                         font_path='tahoma.ttf' if os.path.exists('tahoma.ttf') else None).generate(processed_text)
                    
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title(f'ابر کلمات - {dataset_name} - کلاس {label}')
                    plt.savefig(os.path.join(viz_dir, f'wordcloud_{dataset_name}_class_{label}.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
    
    def _plot_dataset_comparison(self, viz_dir):
        """مقایسه dataset ها"""
        if len(self.datasets) <= 1:
            return
        
        # مقایسه آمار کلی
        stats_data = []
        for dataset_name, stats in self.analysis_results['basic_stats'].items():
            stats_data.append({
                'Dataset': dataset_name,
                'Total Samples': stats['total_samples'],
                'Avg Text Length': stats['text_length']['mean'],
                'Avg Word Count': stats['word_count']['mean'],
                'Vocabulary Size': self.analysis_results['vocabulary'][dataset_name]['unique_words']
            })
        
        df_stats = pd.DataFrame(stats_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # نمودار تعداد نمونه‌ها
        axes[0,0].bar(df_stats['Dataset'], df_stats['Total Samples'])
        axes[0,0].set_title('تعداد نمونه‌ها')
        axes[0,0].set_ylabel('تعداد')
        
        # نمودار میانگین طول متن
        axes[0,1].bar(df_stats['Dataset'], df_stats['Avg Text Length'])
        axes[0,1].set_title('میانگین طول متن')
        axes[0,1].set_ylabel('کاراکتر')
        
        # نمودار میانگین تعداد کلمات
        axes[1,0].bar(df_stats['Dataset'], df_stats['Avg Word Count'])
        axes[1,0].set_title('میانگین تعداد کلمات')
        axes[1,0].set_ylabel('کلمه')
        
        # نمودار اندازه واژگان
        axes[1,1].bar(df_stats['Dataset'], df_stats['Vocabulary Size'])
        axes[1,1].set_title('اندازه واژگان')
        axes[1,1].set_ylabel('کلمه منحصر به فرد')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'dataset_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pattern_analysis(self, viz_dir):
        """تحلیل الگوها"""
        for dataset_name, pattern_data in self.analysis_results['patterns'].items():
            labels = ['کلاس 0', 'کلاس 1']
            cooking_counts = [pattern_data['by_label'][0]['cooking_patterns'],
                             pattern_data['by_label'][1]['cooking_patterns']]
            tech_counts = [pattern_data['by_label'][0]['tech_patterns'],
                          pattern_data['by_label'][1]['tech_patterns']]
            
            x = np.arange(len(labels))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars1 = ax.bar(x - width/2, cooking_counts, width, label='الگوهای آشپزی')
            bars2 = ax.bar(x + width/2, tech_counts, width, label='الگوهای فناوری')
            
            ax.set_xlabel('کلاس')
            ax.set_ylabel('تعداد')
            ax.set_title(f'تحلیل الگوها - {dataset_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            
            # افزودن مقادیر
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'pattern_analysis_{dataset_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_interactive_plots(self):
        """ایجاد نمودارهای تعاملی با Plotly"""
        print("\n🎯 در حال ایجاد نمودارهای تعاملی...")
        
        viz_dir = "interactive_visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # نمودار تعاملی توزیع طول متن
        self._create_interactive_length_plot(viz_dir)
        
        # نمودار تعاملی مقایسه dataset ها
        self._create_interactive_comparison(viz_dir)
        
        # نمودار تعاملی فراوانی کلمات
        self._create_interactive_word_freq(viz_dir)
        
        print(f"✅ نمودارهای تعاملی در پوشه {viz_dir} ذخیره شدند")
    
    def _create_interactive_length_plot(self, viz_dir):
        """نمودار تعاملی توزیع طول متن"""
        fig = make_subplots(rows=1, cols=len(self.datasets),
                           subplot_titles=list(self.datasets.keys()))
        
        for i, (dataset_name, df) in enumerate(self.datasets.items(), 1):
            fig.add_trace(
                go.Histogram(x=df['text_length'], name=f'{dataset_name}',
                           nbinsx=20, opacity=0.7),
                row=1, col=i
            )
        
        fig.update_layout(title_text="توزیع طول متن - نمودار تعاملی",
                         showlegend=True, height=500)
        
        pyo.plot(fig, filename=os.path.join(viz_dir, 'interactive_text_length.html'), 
                auto_open=False)
    
    def _create_interactive_comparison(self, viz_dir):
        """نمودار تعاملی مقایسه dataset ها"""
        if len(self.datasets) <= 1:
            return
        
        stats_data = []
        for dataset_name, stats in self.analysis_results['basic_stats'].items():
            stats_data.append({
                'Dataset': dataset_name,
                'Total_Samples': stats['total_samples'],
                'Avg_Text_Length': round(stats['text_length']['mean'], 2),
                'Avg_Word_Count': round(stats['word_count']['mean'], 2),
                'Unique_Words': self.analysis_results['vocabulary'][dataset_name]['unique_words']
            })
        
        df_stats = pd.DataFrame(stats_data)
        
        # نمودار radar
        fig = go.Figure()
        
        for _, row in df_stats.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Total_Samples']/max(df_stats['Total_Samples'])*100,
                   row['Avg_Text_Length']/max(df_stats['Avg_Text_Length'])*100,
                   row['Avg_Word_Count']/max(df_stats['Avg_Word_Count'])*100,
                   row['Unique_Words']/max(df_stats['Unique_Words'])*100],
                theta=['تعداد نمونه‌ها', 'میانگین طول متن', 'میانگین تعداد کلمات', 'واژگان منحصر'],
                fill='toself',
                name=row['Dataset']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="مقایسه جامع Dataset ها"
        )
        
        pyo.plot(fig, filename=os.path.join(viz_dir, 'interactive_comparison.html'), 
                auto_open=False)
    
    def _create_interactive_word_freq(self, viz_dir):
        """نمودار تعاملی فراوانی کلمات"""
        for dataset_name, vocab_data in self.analysis_results['vocabulary'].items():
            words, frequencies = zip(*vocab_data['most_common_words'][:20])
            
            fig = go.Figure(data=[
                go.Bar(x=list(words), y=list(frequencies),
                      text=list(frequencies), textposition='auto')
            ])
            
            fig.update_layout(
                title=f'کلمات پرتکرار - {dataset_name}',
                xaxis_title='کلمات',
                yaxis_title='فراوانی',
                height=600
            )
            
            pyo.plot(fig, filename=os.path.join(viz_dir, f'interactive_word_freq_{dataset_name}.html'), 
                    auto_open=False)
    
    def generate_comprehensive_report(self):
        """تولید گزارش جامع"""
        print("\n📋 در حال تولید گزارش جامع...")
        
        report = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'datasets_analyzed': list(self.datasets.keys()),
                'total_samples': sum(len(df) for df in self.datasets.values())
            },
            'basic_statistics': self.analysis_results.get('basic_stats', {}),
            'vocabulary_analysis': self.analysis_results.get('vocabulary', {}),
            'pattern_analysis': self.analysis_results.get('patterns', {}),
            'insights': self._generate_insights()
        }
        
        # ذخیره به فرمت JSON
        with open('comprehensive_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # ذخیره گزارش متنی
        self._save_text_report(report)
        
        print("✅ گزارش جامع ذخیره شد:")
        print("   - comprehensive_analysis_report.json")
        print("   - analysis_report.txt")
        
        return report
    
    def _generate_insights(self):
        """تولید بینش‌ها و نتیجه‌گیری"""
        insights = []
        
        # تحلیل توزیع کلاس‌ها
        for dataset_name, stats in self.analysis_results['basic_stats'].items():
            label_dist = stats['label_percentage']
            if abs(label_dist.get(0, 0) - label_dist.get(1, 0)) > 20:
                insights.append(f"❗ در dataset {dataset_name} عدم تعادل کلاس‌ها وجود دارد")
            else:
                insights.append(f"✅ در dataset {dataset_name} کلاس‌ها متعادل هستند")
        
        # تحلیل واژگان
        for dataset_name, vocab in self.analysis_results['vocabulary'].items():
            richness = vocab['vocabulary_richness']
            if richness > 0.7:
                insights.append(f"📚 dataset {dataset_name} دارای واژگان غنی است (تنوع: {richness:.2f})")
            elif richness < 0.3:
                insights.append(f"📖 dataset {dataset_name} دارای واژگان محدود است (تنوع: {richness:.2f})")
        
        # تحلیل الگوها
        for dataset_name, patterns in self.analysis_results['patterns'].items():
            cooking_in_class1 = patterns['by_label'][1]['cooking_patterns']
            tech_in_class0 = patterns['by_label'][0]['tech_patterns']
            
            if cooking_in_class1 > tech_in_class0:
                insights.append(f"🍳 در dataset {dataset_name}: کلاس 1 بیشتر مرتبط با آشپزی است")
            else:
                insights.append(f"💻 در dataset {dataset_name}: کلاس 0 بیشتر مرتبط با فناوری است")
        
        return insights
    
    def _save_text_report(self, report):
        """ذخیره گزارش متنی"""
        with open('analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("گزارش تحلیل جامع داده‌های فارسی\n")
            f.write("Comprehensive Persian Data Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"📅 تاریخ تحلیل: {report['metadata']['analysis_date']}\n")
            f.write(f"📊 تعداد کل نمونه‌ها: {report['metadata']['total_samples']}\n")
            f.write(f"📁 Dataset های تحلیل شده: {', '.join(report['metadata']['datasets_analyzed'])}\n\n")
            
            # آمار پایه
            f.write("🔢 آمار پایه:\n")
            f.write("-" * 30 + "\n")
            for dataset, stats in report['basic_statistics'].items():
                f.write(f"\n{dataset}:\n")
                f.write(f"  • تعداد نمونه‌ها: {stats['total_samples']}\n")
                f.write(f"  • توزیع کلاس‌ها: {stats['label_distribution']}\n")
                f.write(f"  • میانگین طول متن: {stats['text_length']['mean']:.1f} کاراکتر\n")
                f.write(f"  • میانگین تعداد کلمات: {stats['word_count']['mean']:.1f}\n")
            
            # تحلیل واژگان
            f.write(f"\n\n📚 تحلیل واژگان:\n")
            f.write("-" * 30 + "\n")
            for dataset, vocab in report['vocabulary_analysis'].items():
                f.write(f"\n{dataset}:\n")
                f.write(f"  • کل کلمات: {vocab['total_words']}\n")
                f.write(f"  • کلمات منحصر: {vocab['unique_words']}\n")
                f.write(f"  • غنای واژگان: {vocab['vocabulary_richness']:.3f}\n")
                f.write(f"  • کلمات پرتکرار: {[word for word, _ in vocab['most_common_words'][:5]]}\n")
            
            # بینش‌ها
            f.write(f"\n\n💡 بینش‌ها و نتیجه‌گیری:\n")
            f.write("-" * 30 + "\n")
            for insight in report['insights']:
                f.write(f"  {insight}\n")
    
    def run_complete_analysis(self):
        """اجرای تحلیل کامل"""
        print("🚀 شروع تحلیل جامع داده‌های فارسی")
        print("=" * 50)
        
        # بارگذاری داده‌ها
        self.load_data()
        
        if not self.datasets:
            print("❌ هیچ داده‌ای یافت نشد!")
            return
        
        # تحلیل‌های مختلف
        self.basic_statistics()
        self.analyze_vocabulary()
        self.pattern_analysis()
        
        # ایجاد نمودارها
        self.create_visualizations()
        self.create_interactive_plots()
        
        # تولید گزارش
        self.generate_comprehensive_report()
        
        print("\n" + "=" * 50)
        print("✅ تحلیل کامل به پایان رسید!")
        print("\nفایل‌های تولید شده:")
        print("📁 analysis_visualizations/ - نمودارهای استاتیک")
        print("📁 interactive_visualizations/ - نمودارهای تعاملی")
        print("📄 comprehensive_analysis_report.json - گزارش JSON")
        print("📄 analysis_report.txt - گزارش متنی")


def main():
    """تابع اصلی"""
    analyzer = PersianDataAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
