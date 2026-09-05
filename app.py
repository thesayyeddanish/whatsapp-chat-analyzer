        # ========== CHATBOT ==========
        elif section == "🤖 Chatbot":
            st.header("🤖 Chat Analyzer Bot")
            
            st.markdown("""
            ### 💬 Ask Me Anything About Your Chat!
            
            I'm your personal chat assistant. Ask questions or search for specific words!
            """)
            
            # Initialize chat history in session state
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            if "search_results" not in st.session_state:
                st.session_state.search_results = None
            
            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <div style="font-weight: bold; color: #1976d2; margin-bottom: 5px;">👤 You</div>
                        <div>{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div style="font-weight: bold; color: #388e3c; margin-bottom: 5px;">🤖 Bot</div>
                        <div>{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display search results if any
            if st.session_state.search_results is not None and len(st.session_state.search_results) > 0:
                st.markdown("---")
                st.subheader("🔍 Search Results")
                
                # Show results as chat messages
                for idx, row in st.session_state.search_results.iterrows():
                    st.markdown(f"""
                    <div style="background: white; padding: 15px; border-radius: 10px; 
                                margin: 10px 0; border-left: 4px solid #ff9800;
                                box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <div style="font-weight: bold; color: #333;">👤 {row['sender']}</div>
                            <div style="color: #999; font-size: 12px;">📅 {row['date']}</div>
                        </div>
                        <div style="color: #555; line-height: 1.6;">
                        {row['message'][:500]}{'...' if len(str(row['message'])) > 500 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Clear button
                if st.button("🗑️ Clear Search Results"):
                    st.session_state.search_results = None
                    st.rerun()
            
            # Chat input - use form to prevent rerun loop
            st.markdown("---")
            
            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_input(
                    "Type your question or search term:",
                    placeholder="e.g., 'Search for good morning' or 'Who talks more?'",
                    key="chat_input_form"
                )
                
                submitted = st.form_submit_button("Send", type="primary", use_container_width=True)
            
            if submitted and user_input:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Get bot response
                answer, results = chatbot.answer_question(user_input)
                
                # Add bot response to history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                # Store search results if any
                if results is not None and len(results) > 0:
                    st.session_state.search_results = results
                else:
                    st.session_state.search_results = None
                
                # Rerun to show new message
                st.rerun()
            
            # Quick question buttons
            st.markdown("---")
            st.markdown("#### ⚡ Quick Questions:")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📊 Who talks more?", key="q1", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "Who talks more?"})
                    answer, results = chatbot.answer_question("Who talks more?")
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.search_results = None
                    st.rerun()
            
            with col2:
                if st.button("🕐 Peak hour?", key="q2", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "What's our peak hour?"})
                    answer, results = chatbot.answer_question("What's our peak hour?")
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.search_results = None
                    st.rerun()
            
            with col3:
                if st.button("😊 Positive?", key="q3", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "Show positive messages"})
                    answer, results = chatbot.answer_question("Show positive messages")
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.search_results = None
                    st.rerun()
            
            with col4:
                if st.button("🔍 Search 'love'", key="q4", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "Search for love"})
                    answer, results = chatbot.answer_question("Search for love")
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.search_results = results if results is not None and len(results) > 0 else None
                    st.rerun()
            
            # Clear chat button
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.session_state.search_results = None
                    st.rerun()
            
            with col2:
                if st.button("❓ Show Help", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "help"})
                    answer, results = chatbot.answer_question("help")
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.search_results = None
                    st.rerun()
            
            # Help section
            with st.expander("❓ What can I ask?"):
                st.markdown("""
                **🔍 Search Messages:**
                - Search for 'good morning'
                - Find messages with 'love'
                - Show message containing 'meeting'
                - Where did we talk about 'trip'?
                
                **📊 General Stats:**
                - How many total messages?
                - What's our average per day?
                - Who talks more?
                - When's our peak hour?
                
                **😊 Sentiment:**
                - Show me positive messages
                - How many negative messages?
                
                **🔑 Words & Emojis:**
                - What words does [name] use most?
                - What are our top words?
                - How many emojis do we use?
                
                **📱 Messages:**
                - What was the first message?
                - What's the longest message?
                - Show me the last message
                
                **🚀 Behavior:**
                - Who starts conversations more?
                - What's our response time?
                - Who shares more links?
                
                Just type naturally and I'll do my best to answer! 😊
                """)
