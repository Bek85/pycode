
  First, we will need to import the necessary packages for our code:
  
  ```java
  import java.sql.*;
  import java.util.ArrayList;
  import java.util.List;
  ```
  
  Next, we will create a function called "getUsers" that will query the database and return a list of users:
  
  ```java
  public List<String> getUsers() {
    // Initialize the list to store users
    List<String> userList = new ArrayList<>();
    
    // Create a try-catch block to handle any potential errors
    try {
      // Connect to the database
      Connection conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/databaseName", "username", "password");
      
      // Create a statement to execute SQL queries
      Statement stmt = conn.createStatement();
      
      // Execute the query to retrieve all users from the database
      ResultSet rs = stmt.execute("SELECT * FROM users");
      
      // Loop through the result set and add each user to the list
      while(rs.next()) {
        userList.add(rs.getString("username"));
      }
      
      // Close the connection
      conn.close();
      
    } catch (SQLException e) {
      // Handle any potential errors
      e.printStackTrace();
    }
    
   