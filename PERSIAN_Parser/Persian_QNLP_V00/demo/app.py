import streamlit as st
import numpy as np
import random

# -----------------------------------------------------------
# Persian QNLP Demo (Prototype)
# -----------------------------------------------------------

st.set_page_config(page_title="Persian QNLP Demo", layout="centered")

st.title("🧠 Persian QNLP Interactive Demo")
st.markdown("""
این دمو نسخهٔ اولیه‌ای از پروژهٔ QNLP است که ساختار معنایی جمله را بر پایهٔ رویکرد **DisCoCat** نمایش می‌دهد.  
در نسخه‌های بعدی، این بخش به PersianCatParser و مدل QNLP واقعی متصل خواهد شد.
""")

# -----------------------------------------------------------
# User input
# -----------------------------------------------------------
sentence = st.text_input("✍️ یک جمله فارسی بنویس:", "من از هوش مصنوعی خوشم می‌آید")

if st.button("تحلیل جمله"):
    st.subheader("📜 تجزیه نحوی-معنایی (نمونه‌سازی شده):")

    # Mock-up parser output
    tokens = sentence.split()
    parse_tree = " ⟶ ".join(tokens)
    st.write(f"**Parse Tree:** {parse_tree}")

    # Simulated DisCoCat diagram (simple representation)
    st.markdown("**دیاگرام مفهومی (DisCoCat)**")
    st.graphviz_chart(f"""
        digraph G {{
            rankdir=LR;
            {"; ".join([f'"{w}"' for w in tokens])};
            {" -> ".join([f'"{w}"' for w in tokens])};
        }}
    """)

    # Simulated model output
    st.subheader("🧩 نتیجه مدل (نمونه‌سازی شده):")
    sample_outputs = ["مثبت", "منفی", "خنثی"]
    prediction = random.choice(sample_outputs)
    st.success(f"برچسب احساس: **{prediction}**")

# -----------------------------------------------------------
# Footer
# -----------------------------------------------------------
st.markdown("---")
st.caption("Version 0.1 • Developed by Ahmad Shafiei Aporvari • QNLP Research Project")
