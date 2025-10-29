# Edu-Bot---AI-Powered-Learning-Assistant-for-School-Students-
AI-powered educational chatbot for Class X Science built with Streamlit, FAISS, and Google Gemini — answers syllabus-based questions with contextual accuracy and visual hints.
### Sample WPF Chat UI (XAML)

A simple WPF/XAML layout for a desktop chat client that pairs well with the EduBot backend (Streamlit or API). Paste this into a `.xaml` file (e.g., `MainWindow.xaml`) in a WPF project.

```xml
<Window x:Class="EduBotApp.MainWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="Class X EduBot" Height="640" Width="900" WindowStartupLocation="CenterScreen"
        Background="#F7FBFF">
    <Grid Margin="12">
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="*"/>
            <RowDefinition Height="Auto"/>
        </Grid.RowDefinitions>
        <Grid.ColumnDefinitions>
            <ColumnDefinition Width="260"/>
            <ColumnDefinition Width="*"/>
        </Grid.ColumnDefinitions>

        <!-- Sidebar / Brand -->
        <Border Grid.RowSpan="3" Grid.Column="0" Margin="0,0,12,0" CornerRadius="10" Background="#FFFFFF" Padding="14" 
                BorderBrush="#E6EEF6" BorderThickness="1" >
            <StackPanel>
                <Image Source="Assets/AiSPRY_logo.jpg" Height="64" Margin="0,6,0,12" Stretch="Uniform"/>
                <TextBlock Text="EduBot — Class X Science" FontWeight="Bold" FontSize="16" Foreground="#003366" TextWrapping="Wrap"/>
                <TextBlock Text="AI-powered Q&A, FAISS retrieval, Gemini responses" FontSize="12" Opacity="0.8" Margin="0,6,0,12" TextWrapping="Wrap"/>
                <Separator Margin="0,8"/>
                <TextBlock Text="Settings" FontWeight="SemiBold" Margin="0,6"/>
                <StackPanel Orientation="Vertical" Margin="0,8,0,0">
                    <Button Content="Load Knowledge Base" Margin="0,4,0,0" Padding="6"/>
                    <Button Content="Rebuild Index" Margin="0,4,0,0" Padding="6"/>
                    <Button Content="Export Logs" Margin="0,4,0,0" Padding="6"/>
                </StackPanel>
            </StackPanel>
        </Border>

        <!-- Header -->
        <DockPanel Grid.Row="0" Grid.Column="1" LastChildFill="True" Margin="0,0,0,12">
            <StackPanel Orientation="Vertical">
                <TextBlock Text="📘 Class X EduBot" FontSize="20" FontWeight="Bold" Foreground="#003366"/>
                <TextBlock Text="Ask syllabus-based science questions and get contextual answers." FontSize="12" Opacity="0.85"/>
            </StackPanel>
        </DockPanel>

        <!-- Chat area -->
        <Border Grid.Row="1" Grid.Column="1" CornerRadius="8" Background="White" Padding="12" BorderBrush="#E6EEF6" BorderThickness="1">
            <Grid>
                <Grid.RowDefinitions>
                    <RowDefinition Height="*"/>
                    <RowDefinition Height="Auto"/>
                </Grid.RowDefinitions>

                <!-- Messages list -->
                <ScrollViewer Grid.Row="0" VerticalScrollBarVisibility="Auto" Margin="0,0,0,8">
                    <ItemsControl x:Name="MessagesList" >
                        <ItemsControl.ItemTemplate>
                            <DataTemplate>
                                <Border Background="{Binding IsUser, Converter={StaticResource BoolToBrushConverter}}" 
                                        CornerRadius="8" Padding="10" Margin="0,6" MaxWidth="640" HorizontalAlignment="{Binding Alignment}">
                                    <TextBlock Text="{Binding Text}" TextWrapping="Wrap" />
                                </Border>
                            </DataTemplate>
                        </ItemsControl.ItemTemplate>
                    </ItemsControl>
                </ScrollViewer>

                <!-- Input area -->
                <DockPanel Grid.Row="1" LastChildFill="True" Margin="0,6,0,0">
                    <TextBox x:Name="InputBox" Height="44" VerticalContentAlignment="Center" FontSize="14" 
                             AcceptsReturn="True" TextWrapping="Wrap" Margin="0,0,8,0" />
                    <Button x:Name="SendButton" Content="Send" Width="110" Height="44" Padding="8" 
                            HorizontalAlignment="Right" Click="SendButton_Click"/>
                </DockPanel>
            </Grid>
        </Border>

        <!-- Footer / meta -->
        <StackPanel Grid.Row="2" Grid.Column="1" Orientation="Horizontal" HorizontalAlignment="Right" Margin="0,12,0,0">
            <TextBlock Text="Connected to: " VerticalAlignment="Center" Opacity="0.7"/>
            <TextBlock Text="Local EduBot API" FontWeight="Bold" Margin="6,0,0,0"/>
        </StackPanel>
    </Grid>
</Window>
