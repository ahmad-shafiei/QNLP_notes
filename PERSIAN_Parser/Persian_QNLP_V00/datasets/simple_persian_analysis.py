#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تحلیل ساده داده‌های فارسی
Simple Persian Data Analysis Script

اسکریپت تحلیل ساده بدون وابستگی به کتابخانه‌های خارجی
Simple analysis script without external dependencies
"""

import os
import re
import json
from collections import Counter, defaultdict
from datetime import datetime
import math

class SimplePersianAnalyzer:
    def __init__(self, data_directory="."):
        """
        کلاس تحلیل ساده داده‌های فارسی
        Simple Persian Data Analyzer Class
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
                
                self.datasets[dataset_name] = data
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
    
    def calculate_statistics(self, values):
        """محاسبه آمار پایه"""
        if not values:
            return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
        
        sorted_values = sorted(values)
        n = len(values)
        mean = sum(values) / n
        median = sorted_values[n//2] if n % 2 == 1 else (sorted_values[n//2-1] + sorted_values[n//2]) / 2
        variance = sum((x - mean) ** 2 for x in values) / n
        std = math.sqrt(variance)
        
        return {
            'mean': round(mean, 2),
            'median': median,
            'std': round(std, 2),
            'min': min(values),
            'max': max(values)
        }
    
    def basic_statistics(self):
        """محاسبه آمار پایه"""
        print("\n📊 در حال محاسبه آمار پایه...")
        
        stats = {}
        for dataset_name, data in self.datasets.items():
            dataset_stats = {}
            
            # آمار کلی
            labels = [item['label'] for item in data]
            label_counts = Counter(labels)
            total = len(data)
            
            dataset_stats['total_samples'] = total
            dataset_stats['label_distribution'] = dict(label_counts)
            dataset_stats['label_percentage'] = {
                k: round(v * 100.0 / total, 1) for k, v in label_counts.items()
            }
            
            # آمار متن
            text_lengths = [len(item['text']) for item in data]
            word_counts = [len(self.extract_words(item['text'])) for item in data]
            
            dataset_stats['text_length'] = self.calculate_statistics(text_lengths)
            dataset_stats['word_count'] = self.calculate_statistics(word_counts)
            
            stats[dataset_name] = dataset_stats
        
        self.analysis_results['basic_stats'] = stats
        return stats
    
    def analyze_vocabulary(self):
        """تحلیل واژگان"""
        print("\n📚 در حال تحلیل واژگان...")
        
        vocab_analysis = {}
        
        for dataset_name, data in self.datasets.items():
            all_words = []
            words_by_label = {0: [], 1: []}
            
            for item in data:
                words = self.extract_words(item['text'])
                all_words.extend(words)
                words_by_label[item['label']].extend(words)
            
            vocab_stats = {}
            vocab_stats['total_words'] = len(all_words)
            vocab_stats['unique_words'] = len(set(all_words))
            vocab_stats['vocabulary_richness'] = round(
                len(set(all_words)) / len(all_words) if all_words else 0, 3
            )
            
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
        tech_patterns = ['برنامه', 'نرم افزار', 'اپ', 'اجرا', 'اشکالزدایی']
        
        for dataset_name, data in self.datasets.items():
            pattern_stats = {}
            
            # شمارش الگوهای آشپزی و فناوری
            cooking_count = 0
            tech_count = 0
            
            for item in data:
                text = item['text']
                if any(pattern in text for pattern in cooking_patterns):
                    cooking_count += 1
                if any(pattern in text for pattern in tech_patterns):
                    tech_count += 1
            
            pattern_stats['cooking_mentions'] = cooking_count
            pattern_stats['tech_mentions'] = tech_count
            
            # تحلیل الگو بر اساس برچسب
            label_patterns = {}
            for label in [0, 1]:
                label_data = [item for item in data if item['label'] == label]
                label_cooking = sum(1 for item in label_data 
                                  if any(pattern in item['text'] for pattern in cooking_patterns))
                label_tech = sum(1 for item in label_data 
                               if any(pattern in item['text'] for pattern in tech_patterns))
                
                label_patterns[label] = {
                    'cooking_patterns': label_cooking,
                    'tech_patterns': label_tech,
                    'total_samples': len(label_data)
                }
            
            pattern_stats['by_label'] = label_patterns
            patterns[dataset_name] = pattern_stats
        
        self.analysis_results['patterns'] = patterns
        return patterns
    
    def create_ascii_charts(self):
        """ایجاد نمودارهای ASCII ساده"""
        print("\n📈 در حال ایجاد نمودارهای متنی...")
        
        reports_dir = "analysis_reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        with open(os.path.join(reports_dir, 'ascii_charts.txt'), 'w', encoding='utf-8') as f:
            f.write("نمودارهای تحلیل داده‌های فارسی\n")
            f.write("=" * 50 + "\n\n")
            
            # نمودار توزیع برچسب‌ها
            f.write("1. توزیع برچسب‌ها:\n")
            f.write("-" * 30 + "\n")
            
            for dataset_name, stats in self.analysis_results['basic_stats'].items():
                f.write(f"\n{dataset_name}:\n")
                label_dist = stats['label_distribution']
                total = stats['total_samples']
                
                for label, count in label_dist.items():
                    percentage = count * 100 / total
                    bar_length = int(percentage / 2)  # Scale for display
                    bar = "█" * bar_length
                    f.write(f"  کلاس {label}: {bar} {count} ({percentage:.1f}%)\n")
            
            # نمودار کلمات پرتکرار
            f.write(f"\n\n2. کلمات پرتکرار:\n")
            f.write("-" * 30 + "\n")
            
            for dataset_name, vocab in self.analysis_results['vocabulary'].items():
                f.write(f"\n{dataset_name}:\n")
                max_freq = vocab['most_common_words'][0][1] if vocab['most_common_words'] else 1
                
                for word, freq in vocab['most_common_words'][:10]:
                    bar_length = int(freq * 20 / max_freq)  # Scale to max 20 chars
                    bar = "▓" * bar_length
                    f.write(f"  {word:>10}: {bar} {freq}\n")
            
            # نمودار الگوها
            f.write(f"\n\n3. تحلیل الگوها:\n")
            f.write("-" * 30 + "\n")
            
            for dataset_name, patterns in self.analysis_results['patterns'].items():
                f.write(f"\n{dataset_name}:\n")
                
                # نمودار الگوها بر اساس کلاس
                for label in [0, 1]:
                    f.write(f"  کلاس {label}:\n")
                    cooking = patterns['by_label'][label]['cooking_patterns']
                    tech = patterns['by_label'][label]['tech_patterns']
                    total_label = patterns['by_label'][label]['total_samples']
                    
                    if total_label > 0:
                        cooking_pct = cooking * 100 / total_label
                        tech_pct = tech * 100 / total_label
                        
                        cooking_bar = "🍳" * int(cooking_pct / 10)
                        tech_bar = "💻" * int(tech_pct / 10)
                        
                        f.write(f"    آشپزی:  {cooking_bar} {cooking} ({cooking_pct:.1f}%)\n")
                        f.write(f"    فناوری: {tech_bar} {tech} ({tech_pct:.1f}%)\n")
        
        print(f"✅ نمودارهای متنی در {reports_dir}/ascii_charts.txt ذخیره شدند")
    
    def generate_insights(self):
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
                insights.append(f"📚 dataset {dataset_name} دارای واژگان غنی است (تنوع: {richness:.3f})")
            elif richness < 0.3:
                insights.append(f"📖 dataset {dataset_name} دارای واژگان محدود است (تنوع: {richness:.3f})")
        
        # تحلیل الگوها
        for dataset_name, patterns in self.analysis_results['patterns'].items():
            class1_cooking = patterns['by_label'][1]['cooking_patterns']
            class1_total = patterns['by_label'][1]['total_samples']
            class0_tech = patterns['by_label'][0]['tech_patterns']
            class0_total = patterns['by_label'][0]['total_samples']
            
            if class1_total > 0 and class0_total > 0:
                cooking_ratio = class1_cooking / class1_total
                tech_ratio = class0_tech / class0_total
                
                if cooking_ratio > 0.5:
                    insights.append(f"🍳 در dataset {dataset_name}: کلاس 1 قویاً مرتبط با آشپزی است")
                if tech_ratio > 0.5:
                    insights.append(f"💻 در dataset {dataset_name}: کلاس 0 قویاً مرتبط با فناوری است")
        
        return insights
    
    def generate_comprehensive_report(self):
        """تولید گزارش جامع"""
        print("\n📋 در حال تولید گزارش جامع...")
        
        insights = self.generate_insights()
        
        report = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'datasets_analyzed': list(self.datasets.keys()),
                'total_samples': sum(len(data) for data in self.datasets.values())
            },
            'basic_statistics': self.analysis_results.get('basic_stats', {}),
            'vocabulary_analysis': self.analysis_results.get('vocabulary', {}),
            'pattern_analysis': self.analysis_results.get('patterns', {}),
            'insights': insights
        }
        
        # ایجاد پوشه گزارش‌ها
        reports_dir = "analysis_reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # ذخیره به فرمت JSON
        with open(os.path.join(reports_dir, 'comprehensive_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # ذخیره گزارش متنی فارسی
        self.save_persian_report(report, reports_dir)
        
        print("✅ گزارش جامع ذخیره شد:")
        print(f"   - {reports_dir}/comprehensive_report.json")
        print(f"   - {reports_dir}/persian_report.txt")
        
        return report
    
    def save_persian_report(self, report, reports_dir):
        """ذخیره گزارش فارسی"""
        with open(os.path.join(reports_dir, 'persian_report.txt'), 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("گزارش تحلیل جامع داده‌های فارسی\n")
            f.write("Comprehensive Persian Data Analysis Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"📅 تاریخ تحلیل: {report['metadata']['analysis_date']}\n")
            f.write(f"📊 تعداد کل نمونه‌ها: {report['metadata']['total_samples']}\n")
            f.write(f"📁 Dataset های تحلیل شده: {', '.join(report['metadata']['datasets_analyzed'])}\n\n")
            
            # آمار پایه
            f.write("🔢 آمار پایه Dataset ها:\n")
            f.write("=" * 40 + "\n")
            for dataset, stats in report['basic_statistics'].items():
                f.write(f"\n📁 {dataset.upper()}:\n")
                f.write(f"{'':>4}• تعداد نمونه‌ها: {stats['total_samples']}\n")
                f.write(f"{'':>4}• توزیع کلاس‌ها: {stats['label_distribution']}\n")
                f.write(f"{'':>4}• درصد کلاس‌ها: {stats['label_percentage']}\n")
                f.write(f"{'':>4}• میانگین طول متن: {stats['text_length']['mean']} کاراکتر\n")
                f.write(f"{'':>4}• میانگین تعداد کلمات: {stats['word_count']['mean']}\n")
                f.write(f"{'':>4}• دامنه طول متن: {stats['text_length']['min']} - {stats['text_length']['max']}\n")
                f.write(f"{'':>4}• دامنه تعداد کلمات: {stats['word_count']['min']} - {stats['word_count']['max']}\n")
            
            # تحلیل واژگان
            f.write(f"\n\n📚 تحلیل واژگان:\n")
            f.write("=" * 40 + "\n")
            for dataset, vocab in report['vocabulary_analysis'].items():
                f.write(f"\n📁 {dataset.upper()}:\n")
                f.write(f"{'':>4}• کل کلمات: {vocab['total_words']:,}\n")
                f.write(f"{'':>4}• کلمات منحصر به فرد: {vocab['unique_words']:,}\n")
                f.write(f"{'':>4}• غنای واژگان: {vocab['vocabulary_richness']}\n")
                
                f.write(f"{'':>4}• ۱۰ کلمه پرتکرار:\n")
                for i, (word, freq) in enumerate(vocab['most_common_words'][:10], 1):
                    f.write(f"{'':>8}{i:>2}. {word} ({freq} بار)\n")
                
                # تحلیل بر اساس کلاس
                f.write(f"{'':>4}• تحلیل بر اساس کلاس:\n")
                for label, label_vocab in vocab['label_vocabulary'].items():
                    f.write(f"{'':>8}کلاس {label}:\n")
                    f.write(f"{'':>12}○ کل کلمات: {label_vocab['total_words']}\n")
                    f.write(f"{'':>12}○ کلمات منحصر: {label_vocab['unique_words']}\n")
                    f.write(f"{'':>12}○ کلمات پرتکرار: ")
                    top_words = [word for word, _ in label_vocab['most_common'][:5]]
                    f.write(f"{', '.join(top_words)}\n")
            
            # تحلیل الگوها
            f.write(f"\n\n🔍 تحلیل الگوهای متنی:\n")
            f.write("=" * 40 + "\n")
            for dataset, patterns in report['pattern_analysis'].items():
                f.write(f"\n📁 {dataset.upper()}:\n")
                f.write(f"{'':>4}• کل اشارات آشپزی: {patterns['cooking_mentions']}\n")
                f.write(f"{'':>4}• کل اشارات فناوری: {patterns['tech_mentions']}\n")
                
                f.write(f"{'':>4}• تحلیل بر اساس کلاس:\n")
                for label, label_patterns in patterns['by_label'].items():
                    total = label_patterns['total_samples']
                    cooking = label_patterns['cooking_patterns']
                    tech = label_patterns['tech_patterns']
                    
                    cooking_pct = (cooking * 100 / total) if total > 0 else 0
                    tech_pct = (tech * 100 / total) if total > 0 else 0
                    
                    f.write(f"{'':>8}کلاس {label} ({total} نمونه):\n")
                    f.write(f"{'':>12}○ الگوهای آشپزی: {cooking} ({cooking_pct:.1f}%)\n")
                    f.write(f"{'':>12}○ الگوهای فناوری: {tech} ({tech_pct:.1f}%)\n")
            
            # بینش‌ها و نتیجه‌گیری
            f.write(f"\n\n💡 بینش‌ها و نتیجه‌گیری:\n")
            f.write("=" * 40 + "\n")
            for i, insight in enumerate(report['insights'], 1):
                f.write(f"{i:>2}. {insight}\n")
            
            # توصیه‌ها
            f.write(f"\n\n🎯 توصیه‌ها:\n")
            f.write("=" * 40 + "\n")
            
            # تحلیل کیفیت داده‌ها
            total_samples = report['metadata']['total_samples']
            if total_samples < 1000:
                f.write("1. 📈 افزایش حجم داده‌ها برای بهبود دقت مدل توصیه می‌شود\n")
            
            # تحلیل تعادل کلاس‌ها
            imbalanced_datasets = []
            for dataset, stats in report['basic_statistics'].items():
                label_dist = stats['label_percentage']
                if abs(label_dist.get(0, 0) - label_dist.get(1, 0)) > 20:
                    imbalanced_datasets.append(dataset)
            
            if imbalanced_datasets:
                f.write(f"2. ⚖️ تعادل کلاس‌ها در dataset های {', '.join(imbalanced_datasets)} بهبود یابد\n")
            
            # تحلیل الگوها
            clear_patterns = True
            for dataset, patterns in report['pattern_analysis'].items():
                class1_cooking = patterns['by_label'][1]['cooking_patterns']
                class0_tech = patterns['by_label'][0]['tech_patterns']
                class1_total = patterns['by_label'][1]['total_samples'] 
                class0_total = patterns['by_label'][0]['total_samples']
                
                if class1_total > 0 and class0_total > 0:
                    if (class1_cooking / class1_total < 0.7) or (class0_tech / class0_total < 0.7):
                        clear_patterns = False
            
            if clear_patterns:
                f.write("3. ✅ الگوهای کلاس‌بندی واضح هستند - مدل باید عملکرد خوبی داشته باشد\n")
            else:
                f.write("3. 🔄 الگوهای کلاس‌بندی کاملاً واضح نیستند - نیاز به feature engineering\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("پایان گزارش\n")
    
    def run_complete_analysis(self):
        """اجرای تحلیل کامل"""
        print("🚀 شروع تحلیل جامع داده‌های فارسی")
        print("=" * 60)
        
        # بارگذاری داده‌ها
        self.load_data()
        
        if not self.datasets:
            print("❌ هیچ داده‌ای یافت نشد!")
            return
        
        # تحلیل‌های مختلف
        self.basic_statistics()
        self.analyze_vocabulary()
        self.pattern_analysis()
        
        # ایجاد نمودارها و گزارش‌ها
        self.create_ascii_charts()
        self.generate_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("✅ تحلیل کامل به پایان رسید!")
        print("\nفایل‌های تولید شده:")
        print("📁 analysis_reports/ - پوشه گزارش‌ها")
        print("  ├── comprehensive_report.json - گزارش کامل JSON")
        print("  ├── persian_report.txt - گزارش فارسی تفصیلی")
        print("  └── ascii_charts.txt - نمودارهای متنی")
        
        # نمایش خلاصه نتایج
        self.print_summary()
    
    def print_summary(self):
        """نمایش خلاصه نتایج"""
        print("\n📋 خلاصه نتایج:")
        print("-" * 30)
        
        total_samples = sum(len(data) for data in self.datasets.values())
        print(f"📊 کل نمونه‌ها: {total_samples}")
        
        for dataset_name, data in self.datasets.items():
            labels = [item['label'] for item in data]
            label_counts = Counter(labels)
            print(f"📁 {dataset_name}: {len(data)} نمونه - {dict(label_counts)}")
        
        # نمایش کلمات پرتکرار کلی
        all_words = []
        for data in self.datasets.values():
            for item in data:
                all_words.extend(self.extract_words(item['text']))
        
        if all_words:
            top_words = Counter(all_words).most_common(5)
            print(f"🔝 کلمات پرتکرار: {[word for word, _ in top_words]}")


def main():
    """تابع اصلی"""
    analyzer = SimplePersianAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
