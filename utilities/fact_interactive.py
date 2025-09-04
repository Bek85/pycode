import random
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.table import Table
from rich import box
import time

# Initialize Rich console for colorful output with Windows compatibility
console = Console(force_terminal=True, legacy_windows=False)


def print_welcome():
    """Display welcome message with colorful ASCII art"""
    welcome_text = Text()
    welcome_text.append("[*] ", style="bright_blue")
    welcome_text.append("FASCINATING FACTS CHAT", style="bold bright_cyan")
    welcome_text.append(" [*]", style="bright_blue")

    panel = Panel(welcome_text, box=box.DOUBLE, style="bright_cyan", padding=(1, 2))
    console.print(panel)
    console.print()


def print_instructions():
    """Display usage instructions"""
    instructions = Text()
    instructions.append(
        "[>] Ask me anything about the fascinating facts I know!\n\n",
        style="bright_yellow",
    )
    instructions.append("[!] Try asking about:\n", style="bright_green")
    instructions.append("   - Animals (elephants, dolphins, sharks)\n", style="white")
    instructions.append("   - Space & planets (Mars, Venus, Moon)\n", style="white")
    instructions.append(
        "   - Countries & geography (Australia, Russia)\n", style="white"
    )
    instructions.append(
        "   - History & famous people (Einstein, Edison)\n", style="white"
    )
    instructions.append(
        "   - Science & nature (chocolate, honey, water)\n", style="white"
    )
    instructions.append("   - Fun random facts!\n\n", style="white")
    instructions.append("[?] Commands: ", style="bright_magenta")
    instructions.append(
        "'help' for suggestions, 'random' for surprise facts, 'quit' to exit",
        style="bright_white",
    )

    panel = Panel(
        instructions,
        title="How to Use",
        box=box.ROUNDED,
        style="bright_green",
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


def get_random_suggestions():
    """Get random query suggestions"""
    suggestions = [
        "Tell me about space and planets",
        "What's interesting about animals?",
        "Share some geography facts",
        "What about famous inventors?",
        "Tell me weird facts about food",
        "What's fascinating about the human body?",
        "Share some Olympic facts",
        "Tell me about ancient civilizations",
        "What about natural wonders?",
        "Share some technology facts",
    ]
    return random.sample(suggestions, 3)


def display_suggestions():
    """Display random query suggestions"""
    suggestions = get_random_suggestions()

    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("", style="bright_yellow")

    for i, suggestion in enumerate(suggestions, 1):
        table.add_row(f"[?] {suggestion}")

    panel = Panel(
        table, title="[!] Need Ideas? Try These", box=box.ROUNDED, style="bright_yellow"
    )
    console.print(panel)
    console.print()


def get_random_facts(db, num_facts=3):
    """Get random facts from the database"""
    random_queries = [
        "animals",
        "space",
        "food",
        "history",
        "science",
        "geography",
        "technology",
        "nature",
        "human body",
        "sports",
        "art",
        "music",
    ]
    query = random.choice(random_queries)
    results = db.similarity_search(query, k=num_facts)
    return results


def format_results(results, query_type="search"):
    """Format search results with nice styling"""
    if not results:
        console.print(
            "[?] I couldn't find any facts matching your query. Try asking something else!",
            style="bright_red",
        )
        return

    icon = "[*]" if query_type == "random" else "[>]"
    title = "Random Facts" if query_type == "random" else "Here's what I found"

    console.print(f"\n{icon} [bold bright_green]{title}:[/bold bright_green]\n")

    for i, result in enumerate(results, 1):
        fact_text = result.page_content.strip()

        # Create styled text
        styled_fact = Text()
        styled_fact.append(f"  {i}. ", style="bright_cyan bold")
        styled_fact.append(fact_text, style="bright_white")

        # Create panel for each fact
        panel = Panel(styled_fact, box=box.SIMPLE, style="bright_blue", padding=(0, 1))
        console.print(panel)
        time.sleep(0.3)  # Small delay for dramatic effect

    console.print()


def main():
    """Main interactive chat function"""
    try:
        # Load environment variables
        load_dotenv()

        # Show loading message
        console.print(
            "[bold bright_cyan]Loading fascinating facts database...[/bold bright_cyan]"
        )

        # Initialize the embeddings
        embeddings = OpenAIEmbeddings()

        # Initialize the text splitter
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=200, chunk_overlap=0
        )

        # Load and split the documents
        loader = TextLoader("facts.txt")
        documents = loader.load_and_split(text_splitter)

        # Embed and store the documents
        db = Chroma.from_documents(
            documents, embedding=embeddings, persist_directory="emb"
        )

        console.print(
            "[OK] [bold bright_green]Database loaded successfully![/bold bright_green]\n"
        )

        # Display welcome and instructions
        print_welcome()
        print_instructions()
        display_suggestions()

        # Main chat loop
        while True:
            try:
                # Get user input with nice prompt
                query = Prompt.ask(
                    "\n[bold bright_cyan][BOT] What would you like to know?[/bold bright_cyan]",
                    console=console,
                ).strip()

                if not query:
                    continue

                # Handle special commands
                if query.lower() in ["quit", "exit", "bye", "goodbye"]:
                    console.print(
                        "\n[BYE] [bold bright_yellow]Thanks for exploring fascinating facts! Goodbye![/bold bright_yellow]"
                    )
                    break

                elif query.lower() == "help":
                    print_instructions()
                    display_suggestions()
                    continue

                elif query.lower() in ["random", "surprise", "random fact"]:
                    console.print(
                        "[bold bright_cyan]Finding random fascinating facts...[/bold bright_cyan]"
                    )
                    results = get_random_facts(db)
                    format_results(results, "random")
                    continue

                # Search for facts
                console.print(
                    f"[bold bright_cyan]Searching for facts about '{query}'...[/bold bright_cyan]"
                )
                results = db.similarity_search(query, k=4)

                format_results(results)

                # Show suggestions after each query
                if random.random() < 0.3:  # 30% chance to show suggestions
                    console.print(
                        "[dim][TIP] Want more ideas? Type 'help' for suggestions or 'random' for surprise facts![/dim]\n"
                    )

            except KeyboardInterrupt:
                console.print(
                    "\n[BYE] [bold bright_yellow]Goodbye! Thanks for exploring fascinating facts![/bold bright_yellow]"
                )
                break
            except Exception as e:
                console.print(
                    f"[ERR] [bold bright_red]Oops! Something went wrong: {str(e)}[/bold bright_red]"
                )
                console.print("[dim]Try asking something else![/dim]\n")

    except Exception as e:
        console.print(
            f"[ERR] [bold bright_red]Failed to initialize the facts database: {str(e)}[/bold bright_red]"
        )
        console.print(
            "[dim]Make sure you have an OpenAI API key set in your .env file![/dim]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
