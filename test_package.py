#!/usr/bin/env python3
"""Simple test script to verify package structure works."""

def main():
    try:
        # Test config import
        from config import get_llm, list_available_models
        print("Config import successful!")
        print(f"Available models: {list_available_models()}")
        
        # Test that we can get an LLM instance (without calling it)
        try:
            llm = get_llm("remote")
            print("LLM instance created successfully!")
            print(f"LLM type: {type(llm)}")
        except Exception as e:
            print(f"LLM creation failed (expected without API key): {e}")
        
        # Test module imports
        from chat_apps import tchat_gpt
        print("Chat apps module imports work!")
        
        from langchain_demos import multiple_chains
        print("LangChain demos module imports work!")
        
        from utilities import code_test_generator
        print("Utilities module imports work!")
        
        print("\nAll package imports working correctly!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())